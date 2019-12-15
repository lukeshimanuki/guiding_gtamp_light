from PlacePolicy import PlacePolicy
from keras.layers import *
from keras.models import Model
from keras.callbacks import *

import numpy as np
import sys
import tensorflow as tf

INFEASIBLE_SCORE = -sys.float_info.max


# def admon_critic_loss(tau):
def admon_critic_loss(score_data, D_pred):
    # Determine which of Dpred correspond to fake val
    neg_mask = tf.equal(score_data, INFEASIBLE_SCORE)
    y_neg = tf.boolean_mask(D_pred, neg_mask)

    # Determine which of Dpred correspond to true fcn val
    pos_mask = tf.not_equal(score_data, INFEASIBLE_SCORE)
    y_pos = tf.boolean_mask(D_pred, pos_mask)
    score_pos = tf.boolean_mask(score_data, pos_mask)

    # compute mse w.r.t true function values
    mse_on_true_data = K.mean((K.square(score_pos - y_pos)), axis=-1)
    return mse_on_true_data #+ K.mean(y_neg)  # try to minimize the value of y_neg


def uniform_noise(z_size):
    noise_dim = z_size[-1]
    return np.random.uniform([0] * noise_dim, [1] * noise_dim, size=z_size).astype('float32')


def make_repeated_data_for_fake_and_real_samples(data):
    return np.repeat(data, 2, axis=0)


class PlacePolicyAdMon(PlacePolicy):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        self.weight_input = Input(shape=(1,), dtype='float32', name='weight_for_each_sample')
        PlacePolicy.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.critic_output = self.construct_critic()
        self.critic_model = self.construct_critic_model()
        self.policy_loss_model = self.construct_policy_loss_model()

    def construct_critic_model(self):
        disc = Model(inputs=[self.action_input, self.goal_flag_input, self.key_config_input, self.collision_input,
                             self.pose_input],
                     outputs=self.critic_output,
                     name='disc_output')
        disc.compile(loss=admon_critic_loss, optimizer=self.opt_D)
        return disc

    def construct_critic(self):
        raise NotImplementedError

    def construct_policy_output(self):
        raise NotImplementedError

    def construct_policy_loss_model(self):
        for layer in self.critic_model.layers:
            # for some obscure reason, disc weights still get updated when self.disc.fit is called
            # I speculate that this has to do with the status of the layers at the time it was compiled
            layer.trainable = False

        DG_output = self.critic_model([self.policy_output, self.goal_flag_input, self.key_config_input,
                                       self.collision_input, self.pose_input])
        model = Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input,
                              self.noise_input], outputs=[DG_output])
        model.compile(loss=lambda _, x: -tf.reduce_mean(x),
                      optimizer=self.opt_G,
                      metrics=[])

        return model

    def construct_policy_model(self):
        model = Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input,
                              self.noise_input],
                      outputs=[self.policy_output],
                      name='policy_model')
        return model

    def create_callbacks_for_training(self):
        callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10),
            ModelCheckpoint(filepath=self.save_folder + self.weight_file_name,
                            verbose=False,
                            save_best_only=True,
                            save_weights_only=True),
        ]
        return callbacks

    def train_policy(self, states, konf_relevance, poses, rel_konfs, goal_flags, actions, sum_rewards, epochs=500):
        # todo factor this code
        train_idxs, test_idxs = self.get_train_and_test_indices(len(actions))
        train_data, test_data = self.get_train_and_test_data(states, konf_relevance, poses, rel_konfs, goal_flags,
                                                             actions, sum_rewards,
                                                             train_idxs, test_idxs)

        # test data
        t_actions = test_data['actions']
        t_goal_flags = test_data['goal_flags']
        t_poses = test_data['poses']
        t_rel_konfs = test_data['rel_konfs']
        t_collisions = test_data['states']
        t_sum_rewards = test_data['sum_rewards']

        # training data
        actions = train_data['actions']
        goal_flags = train_data['goal_flags']
        poses = train_data['poses']
        rel_konfs = train_data['rel_konfs']
        collisions = train_data['states']
        sum_rewards = train_data['sum_rewards']

        batch_size = 32
        print "Training..."
        for i in range(epochs):
            batch_idxs = range(0, actions.shape[0], batch_size)
            for _ in batch_idxs:
                col_batch, goal_flag_batch, pose_batch, rel_konf_batch, a_batch, sum_reward_batch = \
                    self.get_batch(collisions, goal_flags, poses, rel_konfs, actions, sum_rewards,
                                   batch_size=batch_size)

                states_batch = (goal_flag_batch, rel_konf_batch, col_batch, pose_batch)
                noise_smpls = uniform_noise(z_size=(batch_size, self.dim_noise))
                fake = self.policy_model.predict([goal_flag_batch, rel_konf_batch, col_batch, pose_batch,
                                                  noise_smpls])
                real = a_batch

                # make their scores
                fake_action_q = np.ones((batch_size, 1)) * INFEASIBLE_SCORE  # marks fake data
                real_action_q = sum_reward_batch.reshape((batch_size, 1))
                batch_a = np.vstack([fake, real])
                batch_scores = np.vstack([fake_action_q, real_action_q])

                # train the critic
                self.critic_model.fit([batch_a,
                                       make_repeated_data_for_fake_and_real_samples(goal_flag_batch),
                                       make_repeated_data_for_fake_and_real_samples(rel_konf_batch),
                                       make_repeated_data_for_fake_and_real_samples(col_batch),
                                       make_repeated_data_for_fake_and_real_samples(pose_batch)],
                                      batch_scores,
                                      epochs=1000, verbose=True)
                import pdb;
                pdb.set_trace()

                # train the policy
                self.policy_loss_model.fit([goal_flag_batch, rel_konf_batch, col_batch, pose_batch, noise_smpls],
                                           batch_scores, epochs=1, verbose=False)
