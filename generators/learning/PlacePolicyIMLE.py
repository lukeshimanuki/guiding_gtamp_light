from PlacePolicy import PlacePolicy
from keras.layers import *
from keras.models import Model

import numpy as np
import time
import os
import socket
from functools import partial


if socket.gethostname() == 'lab' or socket.gethostname() == 'phaedra' or socket.gethostname() == 'dell-XPS-15-9560':
    ROOTDIR = './'
else:
    ROOTDIR = '/data/public/rw/pass.port/guiding_gtamp/'


def noise(z_size):
    return np.random.normal(size=z_size).astype('float32')


def custom_loss(y_true, y_pred, weights):
    return K.mean(K.abs(y_true - y_pred) * weights)


class PlacePolicyIMLE(PlacePolicy):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        self.dim_noise = 4
        self.noise_input = Input(shape=(self.dim_noise,), name='noise_input', dtype='float32')
        self.weight_input = Input(shape=(1,), dtype='float32', name='weight_for_each_sample')
        PlacePolicy.__init__(self, dim_action, dim_collision, save_folder, tau, config)

    def construct_policy_output(self):
        raise NotImplementedError

    def construct_policy_model(self):
        model = Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input,
                              self.noise_input, self.weight_input],
                      outputs=[self.policy_output],
                      name='policy_model')

        cl4 = partial(custom_loss, weights=self.weight_input)
        model.compile(loss=cl4, optimizer=self.opt_D)
        return model

    def generate_k_smples_for_multiple_states(self, states, noise_smpls):
        goal_flags, rel_konfs, collisions, poses = states
        k_smpls = []
        k = noise_smpls.shape[1]

        dummy = np.zeros((len(noise_smpls),1))
        for j in range(k):
            actions = self.policy_model.predict([goal_flags, rel_konfs, collisions, poses, noise_smpls[:, j, :], dummy])
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

    def save_weights(self, additional_name=''):
        fdir = ROOTDIR + '/' + self.save_folder + '/'
        fname = self.weight_file_name + additional_name + '.h5'
        if not os.path.isdir(fdir):
            os.makedirs(fdir)
        self.policy_model.save_weights(fdir + fname)

    def load_weights(self):
        fdir = ROOTDIR + '/' + self.save_folder + '/'
        fname = self.weight_file_name + '.h5'
        print "Loading weight ", fdir + fname
        self.policy_model.load_weights(fdir + fname)

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

    def train_policy(self, states, poses, rel_konfs, goal_flags, actions, sum_rewards, epochs=500):
        # todo factor this code
        train_idxs, test_idxs = self.get_train_and_test_indices(len(actions))
        train_data, test_data = self.get_train_and_test_data(states, poses, rel_konfs, goal_flags, actions, sum_rewards,
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

            self.policy_model.fit([goal_flag_batch, rel_konf_batch, col_batch, pose_batch, chosen_noise_smpls, probability_of_being_sampled],
                                  [a_batch],
                                  epochs=100,
                                  validation_data=(
                                      [t_goal_flags, t_rel_konfs, t_collisions, t_poses, t_chosen_noise_smpls, t_probability_of_being_sampled],
                                      [t_actions]),
                                  callbacks=callbacks,
                                  verbose=False)
            after = self.policy_model.get_weights()
            gen_w_norm = np.linalg.norm(np.hstack([(a - b).flatten() for a, b in zip(before, after)]))
            print "Generator weight norm diff", gen_w_norm
            gen_w_norms[epoch % gen_w_norm_patience] = gen_w_norm

            pred = self.policy_model.predict([t_goal_flags, t_rel_konfs, t_collisions, t_poses, t_chosen_noise_smpls])
            valid_err = np.mean(np.linalg.norm(pred - t_actions, axis=-1))
            valid_errs.append(valid_err)

            if valid_err <= np.min(valid_errs):
                self.save_weights()
                patience = 0
            else:
                patience += 1

            if patience > 20:
                break

            print "Val error %.2f patience %d" % (valid_err, patience)
            print np.min(valid_errs)
