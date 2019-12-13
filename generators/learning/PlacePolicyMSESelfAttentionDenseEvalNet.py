from keras.layers import *
from keras.layers.merge import Concatenate
from generators.learning.PlacePolicyMSE import PlacePolicyMSE
from keras.models import Model
from keras import backend as K

import socket
import numpy as np


class PlacePolicyMSESelfAttentionDenseEvalNet(PlacePolicyMSE):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        PlacePolicyMSE.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.weight_file_name = 'place_mse_selfattention_seed_%d' % config.seed
        print "Created PlacePolicyMSESelfAttentionDenseEvalNet"

    def construct_policy_output(self):
        #evalnet_input = Reshape((615, self.dim_action, 1))(candidate_qg)
        eval_net = self.construct_eval_net(0)
        key_config_input = Reshape((615, 4))(self.key_config_input)
        best_qk = Lambda(lambda x: K.batch_dot(x[0], x[1]), name='best_qk')([eval_net, key_config_input])
        output = self.construct_value_output(best_qk)
        return output

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
        concat = Concatenate(axis=-1)([self.pose_input, best_qk])
        dense_num = 32
        value = Dense(dense_num, activation='relu',
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer)(concat)
        value = Dense(dense_num, activation='relu',
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer)(value)
        value = Dense(4, activation='linear',
                      kernel_initializer=self.kernel_initializer,
                      bias_initializer=self.bias_initializer)(value)
        """
        q_0 = self.pose_input
        q_0 = RepeatVector(615)(q_0)
        q_0 = Reshape((615, self.dim_poses, 1))(q_0)
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

        # There currently are 615 candidate goal configurations
        concat_input = Flatten()(self.collision_input)
        dense_num = 8
        collision_feature = Dense(dense_num, activation='relu',
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer)(concat_input)
        collision_feature = Dense(dense_num, activation='relu',
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer)(collision_feature)
        collision_feature = Dense(615, activation='relu',
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer, name='collision_feature')(collision_feature)
        collision_feature = Reshape((615, 1, 1))(collision_feature)

        q_0 = self.pose_input
        q_0 = RepeatVector(615)(q_0)
        q_0 = Reshape((615, self.dim_poses, 1))(q_0)
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
        evalnet = Reshape((615,))(evalnet)

        """
        def get_second_column(x):
            return x[:, :, 1]*100
        col_free_flags = Lambda(get_second_column)(self.collision_input)
        col_free_flags = Reshape((615,))(col_free_flags)
        evalnet = Subtract()([evalnet, col_free_flags])
        """

        def compute_softmax(x):
            return K.softmax(x * 1000, axis=-1)
        evalnet = Lambda(compute_softmax, name='softmax')(evalnet)
        evalnet = Reshape((615,))(evalnet)
        self.evalnet_model = Model(
            inputs=[self.pose_input, self.key_config_input, self.collision_input, self.goal_flag_input],
            outputs=evalnet,
            name='value_model')
        return evalnet

    def construct_policy_model(self):
        mse_model = Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input],
                          outputs=self.policy_output,
                          name='policy_output')
        mse_model.compile(loss='mse', optimizer=self.opt_D)
        return mse_model

    def compute_policy_mse(self, data):
        pred = self.policy_model.predict(
            [data['goal_flags'], data['rel_konfs'], data['states'], data['poses']])
        return np.mean(np.power(pred - data['actions'], 2))
