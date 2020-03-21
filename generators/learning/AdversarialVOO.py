from keras.layers import *
from keras import backend as K
from keras.models import Model
from PlacePolicy import PlacePolicy
from voo.voo import VOO

import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import time
import socket
import os

INFEASIBLE_SCORE = -sys.float_info.max

if socket.gethostname() == 'lab' or socket.gethostname() == 'phaedra' or socket.gethostname() == 'dell-XPS-15-9560':
    ROOTDIR = './'
else:
    ROOTDIR = '/data/public/rw/pass.port/guiding_gtamp_light/learned_weights/'



def make_repeated_data_for_fake_and_real_samples(data):
    return np.repeat(data, 2, axis=0)


def admon_critic_loss(score_data, D_pred):
    # Determine which of Dpred correspond to fake val
    neg_mask = tf.equal(score_data, INFEASIBLE_SCORE)
    y_neg = tf.boolean_mask(D_pred, neg_mask)

    # Determine which of Dpred correspond to true fcn val
    pos_mask = tf.not_equal(score_data, INFEASIBLE_SCORE)
    y_pos = tf.boolean_mask(D_pred, pos_mask)
    #score_pos = tf.boolean_mask(score_data, pos_mask)

    # compute mse w.r.t true function values
    #mse_on_true_data = K.mean((K.square(score_pos - y_pos)), axis=-1)
    return -K.mean(y_pos) + K.mean(y_neg)


class AdversarialVOO(PlacePolicy):
    def __init__(self, dim_action, dim_collision, dim_poses, n_key_configs, save_folder, config):
        PlacePolicy.__init__(self, dim_action, dim_collision, dim_poses, n_key_configs, save_folder, config)
        self.weight_file_name = '%s_adversarial_voo_%d' % (config.atype, config.seed)
        self.value_network_defn = self.construct_eval_net()
        self.value_network = self.construct_model(self.value_network_defn, 'val_network')
        if config.region == 'loading_region':
            domain = np.array([[-0.7, -8.55, -1, -1], [4.3, -4.85, 1, 1]])
        else:
            domain = np.array([[-1.75, -3.16, -1, -1], [5.25, 3.16, 1, 1]])
        self.voo_agent = VOO(domain, 0.3, 'gaussian', None)

    def construct_policy_model(self):
        pass

    def construct_policy_output(self):
        pass

    def sample_from_voo(self, col_batch, goal_flag_batch, pose_batch, konf_batch, voo_iter=30):
        # todo I need to run VOO for each col, goal, pose, and rel konf values

        # For 100 iterations of VOO,
        # Each batch: 0.2129 seconds
        # Total: 6.8 seconds per batch
        # There are 218 batches; Each epoch takes 1482 seconds, which is 24 minutes
        # How can I accelerate this process?
        x_vals_to_return = []
        for goal_flag, key_config, col, pose in zip(goal_flag_batch, konf_batch, col_batch, pose_batch):
            evaled_x = []
            evaled_y = []

            def obj_fcn(x):
                return self.value_network.predict(
                    [x, goal_flag[None, :], key_config[None, :], col[None, :], pose[None, :]])

            for _ in range(voo_iter):
                x = self.voo_agent.sample_next_point(evaled_x, evaled_y)
                evaled_x.append(x)
                val = obj_fcn(np.array([x]))[0, 0]
                evaled_y.append(val)

            x_vals_to_return.append(np.array([evaled_x[np.argmax(evaled_y)]]))

        return np.array(x_vals_to_return).squeeze()

    def construct_model(self, output, name):
        model = Model(inputs=[self.action_input, self.goal_flag_input, self.key_config_input, self.collision_input,
                              self.pose_input],
                      outputs=output,
                      name='critic_output')
        model.compile(loss=admon_critic_loss, optimizer=self.opt_D)
        return model

    def save_weights(self, additional_name=''):
        fdir = ROOTDIR + '/' + self.save_folder + '/'
        fname = self.weight_file_name + additional_name + '.h5'
        if not os.path.isdir(fdir):
            os.makedirs(fdir)
        print "Saving weights", fdir + fname
        self.value_network.save_weights(fdir + fname)

    def load_weights(self, additional_name=''):
        fdir = ROOTDIR + '/' + self.save_folder + '/'
        fname = self.weight_file_name + additional_name + '.h5'
        print "Loading weights", fdir+fname
        self.value_network.load_weights(fdir+fname)

    def train_policy(self, states, poses, rel_konfs, goal_flags, actions, sum_rewards, epochs=500):
        train_idxs, test_idxs = self.get_train_and_test_indices(len(actions))
        train_data, test_data = self.get_train_and_test_data(states, poses, rel_konfs, goal_flags,
                                                             actions, sum_rewards,
                                                             train_idxs, test_idxs)

        # test data
        t_actions = test_data['actions']
        t_goal_flags = test_data['goal_flags']
        t_poses = test_data['poses']
        t_rel_konfs = test_data['rel_konfs']
        t_collisions = test_data['states']
        n_test_data = len(t_actions)
        num_smpl_per_state = 10

        # training data
        actions = train_data['actions']
        goal_flags = train_data['goal_flags']
        poses = train_data['poses']
        rel_konfs = train_data['rel_konfs']
        collisions = train_data['states']
        n_train = len(actions)

        batch_size = 32
        dummy = np.zeros((batch_size, 1))

        valid_errs = []
        patience = 0
        print "Training..."
        g_lr = 1e-4
        d_lr = 1e-3
        self.set_learning_rates(d_lr, g_lr)

        for epoch in range(epochs):
            print  "Epoch %d / %d" %(epoch, epochs)
            batch_idxs = range(0, actions.shape[0], batch_size)
            stime=time.time()
            for batch_idx, _ in enumerate(batch_idxs):
                print "Batch %d / %d" %(batch_idx, len(batch_idxs))
                col_batch, goal_flag_batch, pose_batch, rel_konf_batch, a_batch = \
                    self.get_batch(collisions, goal_flags, poses, rel_konfs, actions,
                                   batch_size=batch_size)
                stime2=time.time()
                fake = self.sample_from_voo(col_batch, goal_flag_batch, pose_batch, rel_konf_batch)
                print "Fake sample generation time", time.time()-stime2
                real = a_batch

                # make their scores
                fake_action_q = np.ones((batch_size, 1)) * INFEASIBLE_SCORE  # marks fake data
                real_action_q = np.ones((batch_size, 1))  # sum_reward_batch.reshape((batch_size, 1))
                batch_a = np.vstack([fake, real])
                batch_scores = np.vstack([fake_action_q, real_action_q])

                # train the critic
                stime3 = time.time()
                self.value_network.fit([batch_a,
                                        make_repeated_data_for_fake_and_real_samples(goal_flag_batch),
                                        make_repeated_data_for_fake_and_real_samples(rel_konf_batch),
                                        make_repeated_data_for_fake_and_real_samples(col_batch),
                                        make_repeated_data_for_fake_and_real_samples(pose_batch)],
                                       batch_scores,
                                       batch_size=64,
                                       epochs=32, verbose=False)
                print "Value network train time", time.time()-stime3
            print time.time()-stime
            self.save_weights('epoch_' + str(epoch))

            """
            t_world_states = (t_goal_flags, t_rel_konfs, t_collisions, t_poses)
            t_noise_smpls = gaussian_noise(z_size=(n_test_data, num_smpl_per_state, self.dim_noise))
            t_generated_actions = self.generate_k_smples_for_multiple_states(t_world_states, t_noise_smpls)
            t_chosen_noise_smpls = self.get_closest_noise_smpls_for_each_action(t_actions, t_generated_actions,
                                                                                t_noise_smpls)
            pred = self.policy_model.predict([t_goal_flags, t_rel_konfs, t_collisions, t_poses,
                                              t_chosen_noise_smpls])
            valid_err = np.mean(np.linalg.norm(pred - t_actions, axis=-1))
            valid_errs.append(valid_err)

            self.save_weights('epoch_' + str(epoch))

            if valid_err <= np.min(valid_errs):
                self.save_weights(additional_name='best_val_err')
                patience = 0
            else:
                patience += 1

            print "Best Val error", np.min(valid_errs)
            print "Val error %.2f patience %d" % (valid_err, patience)
            real_score_values = np.mean(
                (self.critic_model.predict([actions, goal_flags, rel_konfs, collisions, poses])))
            noise_smpls = gaussian_noise(z_size=(n_train, self.dim_noise))
            fake_score_values = np.mean(
                (self.policy_loss_model.predict([goal_flags, rel_konfs, collisions, poses, noise_smpls]).squeeze()))
            print "Real and fake scores %.5f, %.5f" % (real_score_values, fake_score_values)

            # todo gradually reduce the learning rates based on the validation errs
            if real_score_values <= fake_score_values:
                g_lr = 1e-4
                d_lr = 1e-3
            else:
                g_lr = 1e-3
                d_lr = 1e-4

            self.set_learning_rates(d_lr, g_lr)
            """

    def construct_eval_net(self):
        pose_input = RepeatVector(self.n_key_confs)(self.pose_input)
        pose_input = Reshape((self.n_key_confs, self.dim_poses, 1))(pose_input)

        action_input = RepeatVector(self.n_key_confs)(self.action_input)
        action_input = Reshape((self.n_key_confs, self.dim_action, 1))(action_input)

        collision_inp = Flatten()(self.collision_input)
        collision_inp = RepeatVector(self.n_key_confs)(collision_inp)
        collision_inp = Reshape((self.n_key_confs, self.n_collisions * 2, 1))(collision_inp)
        concat_input = Concatenate(axis=2)([pose_input, action_input, collision_inp])
        n_dim = concat_input.shape[2]._value
        dense_num = 32
        H = Conv2D(filters=dense_num,
                   kernel_size=(1, n_dim),
                   strides=(1, 1),
                   activation='relu',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(concat_input)
        H = Conv2D(filters=dense_num,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   activation='relu',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(H)
        H = Conv2D(filters=1,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   activation='relu',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(H)
        H = Flatten()(H)
        H = Dense(dense_num, kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)(H)
        H = Dense(1, kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)(H)
        return H
