from keras.layers import *
from keras.layers.merge import Concatenate
from generators.learning.PlacePolicyMSE import PlacePolicyMSE
from keras.models import Model
from keras import backend as K

import socket
import numpy as np


class PlacePolicyMSESelfAttentionAbsolutePoses(PlacePolicyMSE):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        PlacePolicyMSE.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.weight_file_name = 'place_mse_selfattention_seed_%d' % config.seed
        print "Created PlacePolicyMSESelfAttentionAbsolutePoses"

    def construct_policy_output(self):
        candidate_qg = self.construct_value_output()
        evalnet_input = Reshape((615, 4, 1))(candidate_qg)
        eval_net = self.construct_eval_net(evalnet_input)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]), name='policy_output')([eval_net, candidate_qg])
        return output

    def construct_value_output(self):
        pose_input = RepeatVector(615)(self.pose_input)
        pose_input = Reshape((615, self.dim_poses, 1))(pose_input)
        concat_input = Concatenate(axis=2)([self.key_config_input, pose_input])
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
            inputs=[self.goal_flag_input, self.key_config_input, self.pose_input],
            outputs=value,
            name='value_model')
        return value

    def construct_q0_qg_eval(self, candidate_qg_input):
        pose_input = RepeatVector(615)(self.pose_input)
        pose_input = Reshape((615, self.dim_poses, 1))(pose_input)
        concat_input = Concatenate(axis=2, name='qg_pose')([candidate_qg_input, pose_input])
        n_dim = concat_input.shape[2]._value
        n_filters = 256
        H = Conv2D(filters=n_filters,
                   kernel_size=(1, n_dim),
                   strides=(1, 1),
                   activation='relu',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(concat_input)
        for _ in range(1):
            H = Conv2D(filters=n_filters,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       activation='relu',
                       kernel_initializer=self.kernel_initializer,
                       bias_initializer=self.bias_initializer)(H)
        n_pose_features = 1
        q0_qg_eval = Conv2D(filters=n_pose_features,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            activation='linear',
                            kernel_initializer=self.kernel_initializer,
                            bias_initializer=self.bias_initializer,
                            name='q0_qg_eval')(H)
        def compute_softmax(x):
            return K.softmax(x, axis=-1)
        q0_qg_eval = Lambda(compute_softmax, name='softmax_q0_qg')(q0_qg_eval)
        q0_qg_eval = Reshape((615, 1, 1))(q0_qg_eval)
        #q0_qg_model =  Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input],
        #                  outputs=q0_qg_eval, name='q0qg_model')
        #import pdb;pdb.set_trace()

        return q0_qg_eval

    def construct_eval_net(self, candidate_qg_goal_input):
        # Computes how important each candidate q_g is.
        # w_i = phi_1(x_i)
        # It currently takes in candidate q_g as an input
        # There currently are 615 candidate goal configurations
        q0_qg_eval = self.construct_q0_qg_eval(candidate_qg_goal_input)
        collision_input = Multiply()([q0_qg_eval, self.collision_input])
        concat_input = Flatten()(collision_input)
        dense_num = 8
        evalnet = Dense(dense_num, activation='relu',
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer)(concat_input)
        evalnet = Dense(dense_num, activation='relu',
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer)(evalnet)
        evalnet = Dense(615, activation='linear',
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer)(evalnet)

        def compute_softmax(x):
            return K.softmax(x, axis=-1)

        evalnet = Lambda(compute_softmax, name='softmax')(evalnet)

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
