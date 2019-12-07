from keras.layers import *
from keras.layers.merge import Concatenate
from generators.learning.PlacePolicyMSE import PlacePolicyMSE
from keras.models import Model
from keras import backend as K

import socket
import numpy as np
from PlacePolicyMSESelfAttentionDenseEvalNet import PlacePolicyMSESelfAttentionDenseEvalNet


class PlacePolicyMSESelfAttentionEvalNetWithCandidateGoalAndCollisionInput(PlacePolicyMSESelfAttentionDenseEvalNet):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        PlacePolicyMSESelfAttentionDenseEvalNet.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.weight_file_name = 'place_mse_selfattention_seed_%d' % config.seed
        print "Created PlacePolicyMSESelfAttentionEvalNetWithCandidateGoalAndCollisionInput"

    def construct_eval_net(self, candidate_qg_input):
        collision_input = Flatten()(self.collision_input)
        #collision_q0_input = Concatenate(axis=1)([collision_inputt])
        dense_num = 32
        H = Dense(dense_num, activation='relu',
                  kernel_initializer=self.kernel_initializer,
                  bias_initializer=self.bias_initializer)(collision_input)
        H = Dense(dense_num, activation='relu',
                  kernel_initializer=self.kernel_initializer,
                  bias_initializer=self.bias_initializer)(H)
        collision_summary = Dense(615, activation='linear',
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer)(H)
        collision_summary = Reshape((615, 1, 1), name='collision_summary')(collision_summary)
        pose_input = RepeatVector(615)(self.pose_input)
        pose_input = Reshape((615, 4, 1))(pose_input)
        concat_input = Concatenate(axis=2,  name='candidate_qg_and_collision_summary')([self.key_config_input,
                                                                                        candidate_qg_input,
                                                                                        collision_summary])
        n_dim = concat_input.shape[2].value
        n_filters = 64
        H = Conv2D(filters=n_filters,
                   kernel_size=(1, n_dim),
                   strides=(1, 1),
                   activation='linear',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer,
                   name='evalnet_input')(concat_input)
        H = LeakyReLU()(H)
        for _ in range(2):
            H = Conv2D(filters=n_filters,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       activation='linear',
                       kernel_initializer=self.kernel_initializer,
                       bias_initializer=self.bias_initializer)(H)
            H = LeakyReLU()(H)
        value = Conv2D(filters=1,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       activation='linear',
                       kernel_initializer=self.kernel_initializer,
                       bias_initializer=self.bias_initializer,
                       name='eval_output')(H)

        def compute_softmax(x):
            x = K.squeeze(x, axis=-1)
            x = K.squeeze(x, axis=-1)
            return K.softmax(x, axis=-1)

        evalnet = Lambda(compute_softmax, name='softmax')(value)

        return evalnet
