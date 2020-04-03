from keras.layers import *
from keras import backend as K
from keras.models import Model
from generators.learning.PlacePolicy import PlacePolicy
from generators.learning.voo.voo import VOO
import keras

from gtamp_utils import utils

import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import time
import socket
import os

FAKE_SCORE = sys.float_info.max
INFEASIBLE_SCORE = -sys.float_info.max
FEASIBLE_SCORE = 1

if socket.gethostname() == 'lab' or socket.gethostname() == 'phaedra' or socket.gethostname() == 'dell-XPS-15-9560':
    ROOTDIR = './'
else:
    ROOTDIR = '/data/public/rw/pass.port/guiding_gtamp_light/learned_weights/'


def make_repeated_data_for_fake_and_real_samples(data, n_repeats):
    return np.vstack([data, data]) #np.repeat(data, n_repeats, axis=0)


def admon_critic_loss(score_data, D_pred):
    # Determine which of Dpred correspond to fake val
    neg_mask = tf.equal(score_data, FAKE_SCORE)
    y_neg = tf.boolean_mask(D_pred, neg_mask)

    # Determine which of Dpred correspond to true fcn val
    pos_mask = tf.equal(score_data, FEASIBLE_SCORE)
    y_pos = tf.boolean_mask(D_pred, pos_mask)
    #score_pos = tf.boolean_mask(score_data, pos_mask)

    # infeasible_mask = tf.equal(score_data, INFEASIBLE_SCORE)
    # y_infeasible = tf.boolean_mask(D_pred, infeasible_mask)

    # compute mse w.r.t true function values
    #mse_on_true_data = K.mean((K.square(score_pos - y_pos)), axis=-1)

    # infeasible_loss = 10 * K.mean(K.maximum(1 - (y_pos - y_infeasible), 0.))
    #return mse_on_true_data + K.mean(y_neg)
    #return K.mean(K.maximum(1 - (y_pos-y_neg), 0.))  # + infeasible_loss
    return -K.mean(y_pos) + K.mean(y_neg)


class AdversarialVOO(PlacePolicy):
    def __init__(self, dim_action, dim_collision, dim_poses, n_key_configs, save_folder, config):
        PlacePolicy.__init__(self, dim_action, dim_collision, dim_poses, n_key_configs, save_folder, config)
        self.weight_file_name = '%s_adversarial_voo_dense_hinge_%d' % (config.atype, config.seed)
        self.value_network_defn = self.construct_eval_net()
        self.value_network = self.construct_model(self.value_network_defn, 'val_network')
        if config.region == 'loading_region':
            self.domain = np.array(
                [[-0.34469225, -8.14641946, -1., -0.99999925], [3.92354742, -5.25567767, 1., 0.99999993]])
        else:
            self.domain = np.array(
                [[-1.28392928, -2.95494754, -0.99999998, -0.99999999], [5.01948716, 2.58819546, 1., 1.]])
        self.voo_agent = VOO(self.domain, 0.3, 'gaussian', None)

    def construct_policy_model(self):
        pass

    def construct_policy_output(self):
        pass

    def sample_from_voo(self, col_batch, pose_batch, voo_iter=10,
                        colliding_key_configs=None, tried_samples=[]):
        # todo I need to run VOO for each col, goal, pose, and rel konf values

        # For 100 iterations of VOO,
        # Each batch: 0.2129 seconds
        # Total: 6.8 seconds per batch
        # There are 218 batches; Each epoch takes 1482 seconds, which is 24 minutes
        # How can I accelerate this process?
        x_vals_to_return = []
        n_data = len(col_batch)
        iter = 0
        for col, pose in zip(col_batch, pose_batch):
            # print "Iter %d / %d" % (iter, n_data)
            evaled_x = []
            evaled_y = []

            def obj_fcn(x):
                return self.value_network.predict(
                    [x, col[None, :], pose[None, :]])

            for _ in range(voo_iter):
                x = self.voo_agent.sample_next_point(evaled_x, evaled_y)
                evaled_x.append(x)
                if colliding_key_configs is None and len(tried_samples) == 0:
                    val = obj_fcn(np.array([x]))[0, 0]
                elif len(tried_samples) != 0 and colliding_key_configs is None:
                    xy_of_sample = x[0:2]
                    xys_of_cols = tried_samples[:, 0:2]
                    dists = np.linalg.norm(xy_of_sample - xys_of_cols, axis=-1)
                    val = obj_fcn(np.array([x]))[0, 0] + 100 * max(0.05 - min(dists), 0)
                else:
                    raise NotImplementedError
                    xy_of_sample = x[0:2]
                    xys_of_cols = colliding_key_configs[:, 0:2]
                    dists = np.linalg.norm(xy_of_sample - xys_of_cols, axis=-1)
                    val = obj_fcn(np.array([x]))[0, 0] + 100 * max(0.1 - min(dists), 0)

                evaled_y.append(val)

            x_vals_to_return.append(np.array([evaled_x[np.argmax(evaled_y)]]))
            iter += 1

        return np.array(x_vals_to_return).squeeze()

    def construct_model(self, output, name):
        model = Model(inputs=[self.action_input, self.collision_input, self.pose_input],
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
        print "Loading weights", fdir + fname
        self.value_network.load_weights(fdir + fname)

    def get_infeasible_samples(self, collisions, poses, key_config_idxs, key_configs):
        # key_configs = key_configs[:, key_config_idxs, :].squeeze()[0]
        collisions = collisions[:, key_config_idxs, :].squeeze()
        key_configs = key_configs.squeeze()[:, key_config_idxs, :]
        colliding_idxs = collisions[:, :, 1] == 0
        infeasible_samples = {}
        data_idx = 0
        for idxs, konf in zip(colliding_idxs, key_configs):
            colliding_key_configs = konf[idxs, :]
            infeasible_samples[data_idx] = colliding_key_configs
            data_idx += 1
        return infeasible_samples

    def train_policy(self, states, poses, rel_konfs, goal_flags, actions, sum_rewards, epochs=3000):
        train_idxs, test_idxs = self.get_train_and_test_indices(len(actions))
        train_data, test_data = self.get_train_and_test_data(states, poses, rel_konfs, goal_flags,
                                                             actions, sum_rewards,
                                                             train_idxs, test_idxs)
        key_configs = rel_konfs.squeeze()[0]
        bigger_than_min = np.all(key_configs[:, 0:2] >= self.domain[0, 0:2], axis=1)
        smaller_than_max = np.all(key_configs[:, 0:2] <= self.domain[1, 0:2], axis=1)
        key_config_idxs = bigger_than_min * smaller_than_max
        key_configs = key_configs[key_config_idxs, :]

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

        batch_idxs = range(0, actions.shape[0], batch_size)
        infeasible = self.get_infeasible_samples(collisions, poses, key_config_idxs, rel_konfs)

        callback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=0, mode='auto',
                                                 baseline=None, restore_best_weights=True)

        fake_action_history = []
        value_history = []
        replay_buffer = {data_idx: [] for data_idx in range(n_train)}
        replay_prob = 0.4
        for epoch in range(epochs):
            print "Epoch %d / %d" % (epoch, epochs)
            stime = time.time()
            for idx, batch_idx in enumerate(batch_idxs):
                # print "Batch %d / %d" % (idx, len(batch_idxs))

                col_batch, goal_flag_batch, pose_batch, rel_konf_batch, real_actions, data_idxs = \
                    self.get_batch(collisions, goal_flags, poses, rel_konfs, actions,
                                   batch_size=batch_size,
                                   batch_start_idx=batch_idx)
                # 0.326,0.423 at epoch 100 epoch time epoch time 0.98-1.25830793381
                # 0.594059405940594 at epoch 100 epoch time 0.845104932785 - 1.15
                # These numbers vary across different runs
                fake_actions = []
                for data_idx in data_idxs:
                    past_fake_actions = replay_buffer[data_idx]
                    #if epoch == 300:
                    #    replay_prob = 1
                    rng = np.random.rand()
                    is_time_to_replay = rng > replay_prob and len(past_fake_actions) > 0
                    if is_time_to_replay:
                        """
                        cols_for_fake, poses_for_fake = collisions[data_idx:data_idx + 1, :], poses[
                                                                                              data_idx:data_idx + 1, :]
                        cols_for_fake = np.repeat(cols_for_fake, len(past_fake_actions), axis=0)
                        poses_for_fake = np.repeat(poses_for_fake, len(past_fake_actions), axis=0)
                        past_fake_vals = self.value_network.predict(
                            [np.array(past_fake_actions), cols_for_fake, poses_for_fake])
                        fake_action = past_fake_actions[np.argmax(past_fake_vals)]
                        """
                        fake_action = past_fake_actions[np.random.randint(len(past_fake_actions))]
                    else:
                        new_fake_action = self.sample_from_voo(collisions[data_idx:data_idx + 1, :], poses[data_idx:data_idx + 1, :])
                        replay_buffer[data_idx].append(new_fake_action)
                        fake_action = new_fake_action
                    fake_actions.append(fake_action)
                fake_actions = np.array(fake_actions)

                fake_action_q = np.ones((len(fake_actions), 1)) * FAKE_SCORE  # marks fake data
                real_action_q = np.ones(
                    (len(fake_actions), 1)) * FEASIBLE_SCORE  # sum_reward_batch.reshape((batch_size, 1))

                # batch_a = np.vstack([fake_actions, infeasible_actions, real_actions])
                batch_a = np.vstack([fake_actions, real_actions])
                # batch_scores = np.vstack([fake_action_q, infeasible_action_q, real_action_q])
                batch_scores = np.vstack([fake_action_q, real_action_q])
                # train the critic
                stime3 = time.time()
                repeated_col = make_repeated_data_for_fake_and_real_samples(col_batch, 2)
                repeated_pose = make_repeated_data_for_fake_and_real_samples(pose_batch, 2)
                """
                print self.value_network.evaluate([batch_a, repeated_col, repeated_pose], batch_scores, batch_size=64)
                print (np.maximum(1-(self.value_network.predict([real_actions, col_batch, pose_batch]) -
                                  self.value_network.predict([fake_actions, col_batch, pose_batch])),0)).mean(axis=0)
                """
                self.value_network.fit([batch_a, repeated_col, repeated_pose], batch_scores,
                                       batch_size=64, epochs=1, verbose=False)#, callbacks=[callback])
                # you gain more by assigning low values to the negative actions with MSE+WGAN loss

                #print self.value_network.evaluate([batch_a, repeated_col, repeated_pose], batch_scores, batch_size=64)
                #real_val = self.value_network.predict([real_actions, col_batch, pose_batch])
                #fake_val = self.value_network.predict([fake_actions, col_batch, pose_batch])
                #print (np.maximum(1 - (real_val-fake_val), 0)).mean(axis=0)
                #print (np.maximum(1-(self.value_network.predict([batch_a, repeated_col, repeated_pose])[32:] - self.value_network.predict([batch_a, repeated_col, repeated_pose])[0:32] ),0)).mean(axis=0)

                if data_idxs[0] == 0:
                    fake_action_history.append(fake_actions[0])
                    fake_val = self.value_network.predict([fake_action_history[0][None, :], col_batch[0:1, :], pose_batch[0:1, :]])
                    real_val = self.value_network.predict([real_actions[0:1, :], col_batch[0:1, :], pose_batch[0:1, :]])
                    last_fake_val = self.value_network.predict([fake_action_history[-1][None, :], col_batch[0:1, :], pose_batch[0:1, :]])
                    #print last_fake_val, real_val
                    #value_history.append((fake_val[0,0], real_val[0,0], fake_val[0,0] > real_val[0,0]))
                    value_history.append(fake_val[0, 0] > real_val[0, 0])
                    #print value_history, np.array(value_history).mean(axis=0)

                # print "Value network train time", time.time() - stime3
            print "Epoch time", time.time() - stime

            if epoch % 100 == 0:
                self.save_weights('epoch_' + str(epoch))

            fake_scores = self.value_network.predict([fake_actions, col_batch, pose_batch])
            real_scores = self.value_network.predict([real_actions, col_batch, pose_batch])
            #print self.value_network.evaluate([batch_a, repeated_col, repeated_pose], batch_scores, batch_size=64)
            print "Fake scores", fake_scores.mean()
            print "Real scores", real_scores.mean()

    def construct_eval_net(self):
        dense_num = 32
        """
        collision_inp = Flatten()(self.collision_input)
        concat_input = Concatenate(axis=-1)([self.pose_input, self.action_input, collision_inp])
        H = Dense(dense_num, activation='relu',
                  kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)(concat_input)
        H = Dense(dense_num, activation="relu", kernel_initializer=self.kernel_initializer,
                  bias_initializer=self.bias_initializer)(H)
        H = Dense(1, activation="linear", kernel_initializer=self.kernel_initializer,
                  bias_initializer=self.bias_initializer)(H)
        """
        concat_input = Concatenate(axis=-1)([self.pose_input, self.action_input])
        H = Dense(dense_num, activation='relu',
                  kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)(
            concat_input)
        H = Dense(dense_num, activation='relu',
                  kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)(H)
        H = Dense(618, activation="relu", kernel_initializer=self.kernel_initializer,
                  bias_initializer=self.bias_initializer)(H)
        action_input = Reshape((self.n_key_confs, 1, 1))(H)
        concat_input = Concatenate(axis=2)([action_input, self.collision_input])
        H = Conv2D(filters=dense_num,
                   kernel_size=(1, 3),
                   strides=(1, 1),
                   activation='linear',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(concat_input)
        H = LeakyReLU()(H)
        H = Conv2D(filters=dense_num,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   activation='linear',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(H)
        H = LeakyReLU()(H)
        H = Conv2D(filters=1,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   activation='linear',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(H)
        H = LeakyReLU()(H)
        H = Flatten()(H)
        H = Dense(dense_num, kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)(H)
        H = Dense(1, kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer)(H)

        return H
