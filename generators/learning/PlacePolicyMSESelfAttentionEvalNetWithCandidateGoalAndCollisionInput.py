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

    def construct_policy_output(self):
        # generating candidate q_g
        tiled_pose = self.get_tiled_input(self.pose_input)
        qk_goalflags_input = Concatenate(axis=2)(
            [self.key_config_input, self.goal_flag_input])
        candidate_qg = self.construct_value_output(qk_goalflags_input)

        candidate_qg = Reshape((615, 4, 1))(candidate_qg)
        evalnet_input = Concatenate(axis=2)([candidate_qg, self.goal_flag_input])
        eval_net = self.construct_eval_net(evalnet_input)
        candidate_qg = Reshape((615, 4))(candidate_qg)

        output = Lambda(lambda x: K.batch_dot(x[0], x[1]), name='policy_output')([eval_net, candidate_qg])
        return output

    def construct_eval_net(self, candidate_qg_goal_flag_input):
        # Computes how important each candidate q_g is.
        # w_i = phi_1(x_i)
        # It currently takes in candidate q_g as an input

        # There currently are 615 candidate goal configurations
        collision_input = Flatten()(self.collision_input)
        collision_input = RepeatVector(615)(collision_input)
        collision_input = Reshape((615, 615*2, 1))(collision_input)

        concat_input = Concatenate(axis=2)([candidate_qg_goal_flag_input, collision_input])
        n_dim = concat_input.shape[2].value
        H = Conv2D(filters=32,
                   kernel_size=(1, n_dim),
                   strides=(1, 1),
                   activation='linear',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer,
                   name='evalnet_input')(concat_input)
        H = LeakyReLU()(H)
        for _ in range(2):
            H = Conv2D(filters=32,
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
