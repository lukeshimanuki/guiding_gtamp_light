from keras.layers import *
from keras.layers.merge import Concatenate
from keras import backend as K
from keras.models import Model

from PlacePolicyMSESelfAttentionDenseEvalNet import PlacePolicyMSESelfAttentionDenseEvalNet


class PlacePolicyMSESelfAttentionEvalNetWithCandidateGoalAndCollisionInput(PlacePolicyMSESelfAttentionDenseEvalNet):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        PlacePolicyMSESelfAttentionDenseEvalNet.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.weight_file_name = 'place_mse_selfattention_seed_%d' % config.seed
        print "Created PlacePolicyMSESelfAttentionEvalNetWithCandidateGoalAndCollisionInput"

    def construct_q0_qg_eval(self, candidate_qg_input):
        pose_input = RepeatVector(615)(self.pose_input)
        pose_input = Reshape((615, 20, 1))(pose_input)
        concat_input = Concatenate(axis=2, name='qg_pose')([candidate_qg_input, pose_input])
        n_dim = concat_input.shape[2]._value
        n_filters = 8
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
        n_pose_features = 2
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
        q0_qg_eval = Reshape((615, n_pose_features, 1))(q0_qg_eval)
        return q0_qg_eval

    def construct_eval_net(self, candidate_qg_input):
        q0_qg_eval = self.construct_q0_qg_eval(candidate_qg_input)
        collision_input = Multiply()([self.collision_input, q0_qg_eval])
        collision_input = Flatten()(collision_input)
        dense_num = 8

        # Now what if I learn some features of q0 qg instead?
        collision_input = RepeatVector(615)(collision_input)
        collision_input = Reshape((615, 615*2, 1))(collision_input)
        #concat_input = Concatenate(axis=2, name='evalnet_input')([collision_input, q0_qg_eval])
        concat_input = collision_input
        n_dim = concat_input.shape[2]._value
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
        H = Conv2D(filters=615,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   activation='linear',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(H)

        def take_the_first_row(x):
            return x[:, 0]
        H = Lambda(take_the_first_row, name='take_the_first_row')(H)
        H = Reshape((615,))(H)
        conv_evalnet = H

        def compute_softmax(x):
            return K.softmax(x, axis=-1)

        #evalnet = Multiply()([q0_qg_eval, conv_evalnet])
        evalnet = Lambda(compute_softmax, name='softmax')(conv_evalnet)

        conv_model = Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input], outputs=[evalnet])
        dense_model = Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input], outputs=[evalnet])

        return evalnet
