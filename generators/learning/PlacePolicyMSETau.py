from keras.layers import *
from keras.layers.merge import Concatenate
from generators.learning.PlacePolicyMSE import PlacePolicyMSE
from keras.models import Model
from keras import backend as K

import socket
import numpy as np


def noise(z_size):
    noise_dim = z_size[-1]
    return np.random.uniform([0] * noise_dim, [1] * noise_dim, size=z_size).astype('float32')
    # return np.random.normal(size=z_size).astype('float32')

if socket.gethostname() == 'lab' or socket.gethostname() == 'phaedra':
    ROOTDIR = './'
else:
    ROOTDIR = '/data/public/rw/pass.port/guiding_gtamp/'
import tensorflow as tf


class PlacePolicyMSETau(PlacePolicyMSE):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        PlacePolicyMSE.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.weight_file_name = 'place_mse_qg_combination_seed_%d' % config.seed
        self.loss_model = self.construct_loss_model()

        print "Created Self-attention Dense Gen Net Dense Eval Net"

    def construct_loss_model(self):
        def avg_distance_to_colliding_key_configs(x):
            policy_output = x[0]
            key_configs = x[1]
            diff = policy_output[:, :, 0:2] - key_configs[:, :, 0:2]
            distances = tf.norm(diff, axis=-1)  # ? by 291 by 1

            collisions = x[2]
            collisions = collisions[:, :, 0]
            collisions = tf.squeeze(collisions, axis=-1)
            n_cols = tf.reduce_sum(collisions, axis=1)  # ? by 291 by 1

            hinge_on_given_dist_limit = tf.maximum(1 - distances, 0)
            hinged_dists_to_colliding_configs = tf.multiply(hinge_on_given_dist_limit, collisions)
            return tf.reduce_sum(hinged_dists_to_colliding_configs, axis=-1) / n_cols

        repeated_poloutput = RepeatVector(self.n_key_confs)(self.policy_output)
        konf_input = Reshape((self.n_key_confs, 4))(self.key_config_input)
        diff_output = Lambda(avg_distance_to_colliding_key_configs, name='collision_distance_output')(
            [repeated_poloutput, konf_input, self.collision_input])

        model = Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input,
                              self.noise_input],
                      outputs=[self.policy_output, self.evalnet_presum],
                      name='loss_model')

        def custom_mse(y_true, y_pred):
            return tf.reduce_mean(tf.norm(y_true - y_pred, axis=-1))

        def cross_entropy_on_tau(y_true, y_pred):
            """
            neg_mask = tf.equal(y_true, 0)
            y_neg = tf.boolean_mask(y_true, neg_mask)
            y_pred_neg = tf.boolean_mask(y_pred, neg_mask)

            pos_mask = tf.equal(y_true, 1)
            y_pos = tf.boolean_mask(y_true, pos_mask)
            y_pred_pos = tf.boolean_mask(y_pred, pos_mask)

            neg_loss = -tf.log(1-y_pred_neg)
            pos_loss = -tf.log(y_pred_pos)
            return tf.reduce_mean((neg_loss+pos_loss))
            """
            return -tf.reduce_sum(y_true * tf.log(y_pred), -1)

        # model.compile(loss=[lambda _, pred: pred, 'mse'], optimizer=self.opt_D, loss_weights=[0, 1])
        model.compile(loss=['mse', 'categorical_crossentropy'], optimizer=self.opt_D, loss_weights=[1, 0.01])
        return model

    def construct_policy_output(self):
        candidate_qg = self.construct_value_output()
        eval_net = self.construct_eval_net()
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]), name='policy_output')([eval_net, candidate_qg])
        return output

    def construct_value_output(self):
        pose_input = RepeatVector(self.n_key_confs)(self.pose_input)
        pose_input = Reshape((self.n_key_confs, self.dim_poses, 1))(pose_input)

        concat_input = Concatenate(axis=2)([pose_input, self.key_config_input])

        n_dim = concat_input.shape[2]._value
        n_filters = 32
        H = Conv2D(filters=n_filters,
                   kernel_size=(1, n_dim),
                   strides=(1, 1),
                   activation='relu',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(concat_input)
        for _ in range(2):
            H = Conv2D(filters=n_filters,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       activation='relu',
                       kernel_initializer=self.kernel_initializer,
                       bias_initializer=self.bias_initializer)(H)
        value = Conv2D(filters=4,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       activation='linear',
                       kernel_initializer=self.kernel_initializer,
                       bias_initializer=self.bias_initializer)(H)

        value = Lambda(lambda x: K.squeeze(x, axis=2), name='candidate_qg')(value)
        self.value_model = Model(
            inputs=[self.goal_flag_input, self.key_config_input, self.pose_input, self.noise_input],
            outputs=value,
            name='value_model')
        return value

    def construct_eval_net(self):
        collision_input = Flatten()(self.collision_input)
        concat_input = Concatenate(axis=1, name='q0_ck')([self.pose_input, collision_input])
        dense_num = 32
        evalnet = Dense(dense_num, activation='relu',
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer)(concat_input)
        evalnet = Dense(dense_num, activation='relu',
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer)(evalnet)
        evalnet = Dense(self.n_key_confs, activation='sigmoid',
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer, name='collision_feature')(evalnet)
        evalnet = Reshape((self.n_key_confs,), name='path_relevance')(evalnet)
        self.evalnet_presum = evalnet
        self.evalnet_presum_model = self.construct_model(evalnet, 'enetmodel')

        def compute_sum(x):
            return K.sum(x, axis=-1)

        summation = Lambda(compute_sum, name='softmax')(evalnet)
        summation = Reshape((1,))(summation)
        summation = RepeatVector(self.n_key_confs)(summation)
        summation = Reshape((self.n_key_confs,))(summation)
        evalnet = Lambda(lambda x: x[0] / x[1])([evalnet, summation])

        """
        def compute_softmax(x):
            return K.softmax(x*100, axis=-1)
        evalnet = Lambda(compute_softmax, name='softmax')(evalnet)
        """

        return evalnet

    def construct_policy_model(self):
        mse_model = Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input,
                                  self.noise_input],
                          outputs=self.policy_output,
                          name='policy_output')
        mse_model.compile(loss='mse', optimizer=self.opt_D)
        return mse_model

    def train_policy(self, states, konf_relevance, poses, rel_konfs, goal_flags, actions, sum_rewards, epochs=500):
        train_idxs, test_idxs = self.get_train_and_test_indices(len(actions))
        train_data, test_data = self.get_train_and_test_data(states, konf_relevance, poses, rel_konfs, goal_flags,
                                                             actions, sum_rewards,
                                                             train_idxs, test_idxs)
        callbacks = self.create_callbacks_for_training()

        actions = train_data['actions']
        goal_flags = train_data['goal_flags']
        poses = train_data['poses']
        rel_konfs = train_data['rel_konfs']
        collisions = train_data['states']
        konf_relevance = train_data['konf_relevance']
        noise_smpls = noise(z_size=(len(actions), self.dim_noise))
        inp = [goal_flags, rel_konfs, collisions, poses, noise_smpls]

        """
        evalnet_presum = self.evalnet_presum_model.predict(inp)[0]
        true = konf_relevance[0]
        evaluated = self.evalnet_presum_model.evaluate([goal_flags[0:1], rel_konfs[0:1], collisions[0:1], poses[0:1], noise_smpls[0:1]], konf_relevance[0:1])
        import pdb;pdb.set_trace()
        """
        pre_mse = self.compute_policy_mse(test_data)
        self.loss_model.fit(inp, [actions, konf_relevance],
                            batch_size=32,
                            epochs=epochs,
                            verbose=2,
                            callbacks=callbacks,
                            validation_split=0.1, shuffle=False)
        # load the best model
        self.load_weights()
        post_mse = self.compute_policy_mse(test_data)
        print "Pre-and-post test errors", pre_mse, post_mse
