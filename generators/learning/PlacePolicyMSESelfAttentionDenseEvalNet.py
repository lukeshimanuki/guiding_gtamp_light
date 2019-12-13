from keras.layers import *
from keras.layers.merge import Concatenate
from generators.learning.PlacePolicyMSE import PlacePolicyMSE
from keras.models import Model
from keras import backend as K

import socket
import numpy as np
import tensorflow as tf

def noise(z_size):
    return np.random.normal(size=z_size).astype('float32')


class PlacePolicyMSEBestqkTransformation(PlacePolicyMSE):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        PlacePolicyMSE.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.loss_model = self.construct_loss_model()
        self.weight_file_name = 'place_mse_selfattention_seed_%d' % config.seed
        print "Created PlacePolicyMSESelfAttentionDenseEvalNet"

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
        noise_smpls = noise(z_size=(len(actions), self.dim_noise))

        inp = [goal_flags, rel_konfs, collisions, poses, noise_smpls]
        pre_mse = self.compute_policy_mse(test_data)
        self.loss_model.fit(inp, [actions, actions],
                            batch_size=32,
                            epochs=epochs,
                            verbose=2,
                            callbacks=callbacks,
                            validation_split=0.1, shuffle=False)
        # load the best model
        self.load_weights()
        post_mse = self.compute_policy_mse(test_data)
        print "Pre-and-post test errors", pre_mse, post_mse

    def construct_policy_output(self):
        # evalnet_input = Reshape((self.n_key_confs, self.dim_action, 1))(candidate_qg)
        eval_net = self.construct_eval_net(0)
        key_config_input = Reshape((self.n_key_confs, 4))(self.key_config_input)
        best_qk = Lambda(lambda x: K.batch_dot(x[0], x[1]), name='best_qk')([eval_net, key_config_input])
        self.best_qk_model = self.construct_model(best_qk, 'best_qk')
        output = self.construct_value_output(best_qk)
        return output

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

        model = Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input, self.noise_input],
                      outputs=[diff_output, self.policy_output],
                      name='loss_model')

        model.compile(loss=[lambda _, pred: pred, 'mse'], optimizer=self.opt_D, loss_weights=[1, 1])
        return model

    def construct_model(self, output, name):
        model = Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input],
                      outputs=[output],
                      name=name)
        return model

    def construct_value_output(self, best_qk):
        # Computes the candidate goal configurations
        # q_g = phi_2(x_i), for some x_i
        # todo: change the create_conv_layers activation to relu for generating value output.
        # value = self.create_conv_layers(concat_input, n_filters=128,
        #                                use_pooling=False, use_flatten=False)
        concat = Concatenate(axis=-1)([self.pose_input, best_qk, self.noise_input])
        dense_num = 32
        value = Dense(dense_num, activation='relu',
                      kernel_initializer=self.kernel_initializer,
                      bias_initializer=self.bias_initializer)(concat)
        value = Dense(4, activation='linear',
                      kernel_initializer=self.kernel_initializer,
                      bias_initializer=self.bias_initializer, name='policy_ouput')(value)
        """
        q_0 = self.pose_input
        q_0 = RepeatVector(self.n_key_confs)(q_0)
        q_0 = Reshape((self.n_key_confs, self.dim_poses, 1))(q_0)
        key_config_input = self.key_config_input
        concat_input = Concatenate(axis=2, name='q0_qk_ck')([q_0, key_config_input, self.collision_input])
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
            inputs=[self.pose_input, self.key_config_input, self.collision_input, self.goal_flag_input],
            outputs=value,
            name='value_model')
        """
        return value

    def construct_eval_net(self, candidate_qg):
        # Computes how important each candidate q_g is.
        # w_i = phi_1(x_i)
        # It currently takes in candidate q_g as an input

        # There currently are self.n_key_confs candidate goal configurations
        collision_input = Flatten()(self.collision_input)
        concat_input = Concatenate(axis=1, name='q0_ck')([self.pose_input, collision_input])

        evalnet = Dense(64, activation='relu',
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer)(concat_input)
        evalnet = Dense(32, activation='relu',
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer)(evalnet)
        evalnet = Dense(self.n_key_confs, activation='linear',
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer, name='collision_feature')(evalnet)
        evalnet = Reshape((self.n_key_confs,))(evalnet)

        """
        q_0 = self.pose_input
        q_0 = RepeatVector(self.n_key_confs)(q_0)
        q_0 = Reshape((self.n_key_confs, self.dim_poses, 1))(q_0)
        concat_input = Concatenate(axis=2, name='q0_qg_qk_ck')([q_0, self.key_config_input, self.collision_input])
        #concat_input = Concatenate(axis=2, name='q0_qg_qk_ck')([q_0, candidate_qg])
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
        evalnet = Conv2D(filters=1,
                         kernel_size=(1, 1),
                         strides=(1, 1),
                         activation='linear',
                         kernel_initializer=self.kernel_initializer,
                         bias_initializer=self.bias_initializer)(H)
        evalnet = Reshape((self.n_key_confs,))(evalnet)
        """

        def get_first_column(x):
            return x[:, :, 0] * 100

        col_free_flags = Lambda(get_first_column)(self.collision_input)
        col_free_flags = Reshape((self.n_key_confs,))(col_free_flags)
        evalnet = Subtract()([evalnet, col_free_flags])

        def compute_softmax(x):
            return K.softmax(x * 100, axis=-1)

        evalnet = Lambda(compute_softmax, name='softmax')(evalnet)
        evalnet = Reshape((self.n_key_confs,))(evalnet)
        self.evalnet_model = Model(
            inputs=[self.pose_input, self.key_config_input, self.collision_input, self.goal_flag_input],
            outputs=evalnet,
            name='value_model')
        return evalnet

    def construct_policy_model(self):
        # noise input is used to make the prediction format consistent with imle
        mse_model = Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input,
                                  self.noise_input],
                          outputs=self.policy_output,
                          name='policy_output')
        mse_model.compile(loss='mse', optimizer=self.opt_D)
        return mse_model

    def compute_policy_mse(self, data):
        dummy = np.zeros((len(data['goal_flags']), 4))
        pred = self.policy_model.predict(
            [data['goal_flags'], data['rel_konfs'], data['states'], data['poses'], dummy])
        return np.mean(np.power(pred - data['actions'], 2))
