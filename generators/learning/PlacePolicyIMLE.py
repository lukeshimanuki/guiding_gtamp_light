from PlacePolicy import PlacePolicy
from keras.layers import *
from keras.models import Model
from keras.callbacks import *

import numpy as np
import time
import tensorflow as tf


def noise(z_size):
    return np.random.normal(size=z_size).astype('float32')


class PlacePolicyIMLE(PlacePolicy):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        self.weight_input = Input(shape=(1,), dtype='float32', name='weight_for_each_sample')
        PlacePolicy.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.loss_model = self.construct_loss_model()

    def construct_policy_output(self):
        raise NotImplementedError

    def construct_policy_model(self):
        model = Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input,
                              self.noise_input],
                      outputs=[self.policy_output],
                      name='policy_model')
        model.compile(loss='mse', optimizer=self.opt_D)
        return model

    def construct_loss_model(self):
        def avg_distance_to_colliding_key_configs(x):
            policy_output = x[0]
            key_configs = x[1]
            diff = policy_output - key_configs
            distances = tf.norm(diff, axis=-1)  # ? by 291 by 1

            collisions = x[2]
            collisions = collisions[:, :, 0]
            collisions = tf.squeeze(collisions, axis=-1)
            n_cols = tf.reduce_sum(collisions, axis=1)  # ? by 291 by 1

            hinge_on_given_dist_limit = tf.maximum(1-distances, 0)
            hinged_dists_to_colliding_configs = tf.multiply(hinge_on_given_dist_limit, collisions)
            return tf.reduce_sum(hinged_dists_to_colliding_configs, axis=-1) / n_cols

        repeated_poloutput = RepeatVector(self.n_key_confs)(self.policy_output)
        konf_input = Reshape((self.n_key_confs, 4))(self.key_config_input)
        diff_output = Lambda(avg_distance_to_colliding_key_configs, name='collision_distance_output')(
            [repeated_poloutput, konf_input, self.collision_input])

        model = Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input,
                              self.noise_input],
                      outputs=[diff_output, self.policy_output],
                      name='loss_model')

        model.compile(loss=[lambda _, pred: pred, 'mse'], optimizer=self.opt_D, loss_weights=[1, 2])
        return model

    def generate_k_smples_for_multiple_states(self, states, noise_smpls):
        goal_flags, rel_konfs, collisions, poses = states
        k_smpls = []
        k = noise_smpls.shape[1]

        dummy = np.zeros((len(noise_smpls), 1))
        for j in range(k):
            actions = self.loss_model.predict([goal_flags, rel_konfs, collisions, poses, noise_smpls[:, j, :]])
            #import pdb;pdb.set_trace()
            #idx = 399
            #first = actions[idx]
            #action_output = self.real_policy_model.predict([goal_flags, rel_konfs, collisions, poses, noise_smpls[:, j, :]])
            #colliding_dists = np.linalg.norm(action_output[idx].squeeze() - rel_konfs[1].squeeze(), axis=-1) * collisions[idx][:, 0].squeeze()
            #colliding_dists = np.maximum(colliding_dists - 0.1, 0)
            #print np.sum(colliding_dists) / np.sum(collisions[idx][:, 0].squeeze())
            #print first
            #import pdb;pdb.set_trace()
            k_smpls.append(actions)
        new_k_smpls = np.array(k_smpls).swapaxes(0, 1)
        return new_k_smpls

    @staticmethod
    def find_the_idx_of_closest_point_to_x1(x1, database):
        l2_distances = np.linalg.norm(x1 - database, axis=-1)
        return database[np.argmin(l2_distances)], np.argmin(l2_distances)

    def get_closest_noise_smpls_for_each_action(self, actions, generated_actions, noise_smpls):
        chosen_noise_smpls = []
        for true_action, generated, noise_smpls_for_action in zip(actions, generated_actions, noise_smpls):
            closest_point, closest_point_idx = self.find_the_idx_of_closest_point_to_x1(true_action, generated)
            noise_that_generates_closest_point_to_true_action = noise_smpls_for_action[closest_point_idx]
            chosen_noise_smpls.append(noise_that_generates_closest_point_to_true_action)
        return np.array(chosen_noise_smpls)

    @staticmethod
    def get_batch_based_on_rewards(cols, goal_flags, poses, rel_konfs, actions, sum_rewards, batch_size):
        indices = np.random.randint(0, actions.shape[0], size=batch_size)

        n_data = actions.shape[0]
        probability_of_being_sampled = np.exp(sum_rewards) / np.sum(np.exp(sum_rewards))
        indices = np.random.choice(n_data, batch_size, p=probability_of_being_sampled)
        cols_batch = np.array(cols[indices, :])  # collision vector
        goal_flag_batch = np.array(goal_flags[indices, :])  # collision vector
        a_batch = np.array(actions[indices, :])
        pose_batch = np.array(poses[indices, :])
        konf_batch = np.array(rel_konfs[indices, :])
        sum_reward_batch = np.array(sum_rewards[indices, :])
        return cols_batch, goal_flag_batch, pose_batch, konf_batch, a_batch, sum_reward_batch

    def create_callbacks_for_training(self):
        callbacks = [
            TerminateOnNaN(),
            # EarlyStopping(monitor='val_loss',min_delta=1e-4,patience=10),
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

        t_actions = test_data['actions']
        t_goal_flags = test_data['goal_flags']
        t_poses = test_data['poses']
        t_rel_konfs = test_data['rel_konfs']
        t_collisions = test_data['states']
        t_sum_rewards = test_data['sum_rewards']

        n_test_data = len(t_collisions)

        data_resampling_step = 1
        num_smpl_per_state = 10

        actions = train_data['actions']
        goal_flags = train_data['goal_flags']
        poses = train_data['poses']
        rel_konfs = train_data['rel_konfs']
        collisions = train_data['states']
        sum_rewards = train_data['sum_rewards']
        callbacks = self.create_callbacks_for_training()

        gen_w_norm_patience = 10
        gen_w_norms = [-1] * gen_w_norm_patience
        valid_errs = []
        patience = 0
        for epoch in range(epochs):
            print 'Epoch %d/%d' % (epoch, epochs)
            is_time_to_smpl_new_data = epoch % data_resampling_step == 0
            batch_size = 400
            col_batch, goal_flag_batch, pose_batch, rel_konf_batch, a_batch, sum_reward_batch = \
                self.get_batch(collisions, goal_flags, poses, rel_konfs, actions, sum_rewards, batch_size=batch_size)

            if is_time_to_smpl_new_data:
                stime = time.time()
                # train data
                world_states = (goal_flag_batch, rel_konf_batch, col_batch, pose_batch)
                noise_smpls = noise(z_size=(batch_size, num_smpl_per_state, self.dim_noise))
                generated_actions = self.generate_k_smples_for_multiple_states(world_states, noise_smpls)

                chosen_noise_smpls = self.get_closest_noise_smpls_for_each_action(a_batch, generated_actions,
                                                                                  noise_smpls)

                # validation data
                t_world_states = (t_goal_flags, t_rel_konfs, t_collisions, t_poses)
                t_noise_smpls = noise(z_size=(n_test_data, num_smpl_per_state, self.dim_noise))
                t_generated_actions = self.generate_k_smples_for_multiple_states(t_world_states, t_noise_smpls)
                t_chosen_noise_smpls = self.get_closest_noise_smpls_for_each_action(t_actions, t_generated_actions,
                                                                                    t_noise_smpls)

                print "Data generation time", time.time() - stime

            # I also need to tag on the Q-learning objective
            before = self.policy_model.get_weights()
            probability_of_being_sampled = np.exp(sum_reward_batch) / np.sum(np.exp(sum_reward_batch))
            t_probability_of_being_sampled = np.exp(t_sum_rewards) / np.sum(np.exp(t_sum_rewards))

            self.loss_model.fit([goal_flag_batch, rel_konf_batch, col_batch, pose_batch, chosen_noise_smpls],
                                [a_batch, a_batch],
                                epochs=200,
                                validation_data=(
                                    [t_goal_flags, t_rel_konfs, t_collisions, t_poses, t_chosen_noise_smpls],
                                    [t_actions, t_actions]),
                                callbacks=callbacks,
                                verbose=False)
            # self.load_weights()
            after = self.policy_model.get_weights()
            gen_w_norm = np.linalg.norm(np.hstack([(a - b).flatten() for a, b in zip(before, after)]))
            print "Generator weight norm diff", gen_w_norm
            gen_w_norms[epoch % gen_w_norm_patience] = gen_w_norm

            pred = self.policy_model.predict([t_goal_flags, t_rel_konfs, t_collisions, t_poses, t_chosen_noise_smpls])
            valid_err = np.mean(np.linalg.norm(pred - t_actions, axis=-1))
            valid_errs.append(valid_err)
            self.save_weights()

            if valid_err <= np.min(valid_errs):
                self.save_weights()
                patience = 0
            else:
                patience += 1

            if patience > 10:
                pass

            print "Val error %.2f patience %d" % (valid_err, patience)
            print np.min(valid_errs)
