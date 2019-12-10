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
        pose_input = Reshape((615, self.dim_poses, 1))(pose_input)
        concat_input = Concatenate(axis=2, name='qg_pose')([candidate_qg_input, pose_input])
        n_dim = concat_input.shape[2]._value
        n_filters = 32
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
            x = K.squeeze(x, axis=-1)
            x = K.squeeze(x, axis=-1)
            return K.softmax(x, axis=-1)
        q0_qg_eval = Lambda(compute_softmax, name='softmax_q0_qg')(q0_qg_eval)
        q0_qg_eval = Reshape((615,1,1))(q0_qg_eval)
        return q0_qg_eval

    def construct_eval_net(self, candidate_qg_input):
        q0_qg_eval = self.construct_q0_qg_eval(candidate_qg_input)
        collision_input = Multiply()([q0_qg_eval, self.collision_input])
        collision_input = Flatten()(collision_input)

        dense_num = 8
        evalnet = Dense(dense_num, activation='relu',
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer)(collision_input)
        evalnet = Dense(dense_num, activation='relu',
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer)(evalnet)
        evalnet = Dense(615, activation='linear',
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer)(evalnet)

        def compute_softmax(x):
            return K.softmax(x, axis=-1)

        evalnet = Add()([q0_qg_eval, evalnet])
        evalnet = Lambda(compute_softmax, name='softmax')(evalnet)

        return evalnet
