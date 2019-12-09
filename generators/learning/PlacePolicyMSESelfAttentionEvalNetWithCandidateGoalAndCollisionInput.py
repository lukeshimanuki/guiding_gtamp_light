from keras.layers import *
from keras.layers.merge import Concatenate
from keras import backend as K

from PlacePolicyMSESelfAttentionDenseEvalNet import PlacePolicyMSESelfAttentionDenseEvalNet


class PlacePolicyMSESelfAttentionEvalNetWithCandidateGoalAndCollisionInput(PlacePolicyMSESelfAttentionDenseEvalNet):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        PlacePolicyMSESelfAttentionDenseEvalNet.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.weight_file_name = 'place_mse_selfattention_seed_%d' % config.seed
        print "Created PlacePolicyMSESelfAttentionEvalNetWithCandidateGoalAndCollisionInput"

    def construct_candidate_qg_from_q0_eval(self, candidate_qg_input):
        pose_input = RepeatVector(615)(self.pose_input)
        pose_input = Reshape((615, 4, 1))(pose_input)
        concat_input = Concatenate(axis=2)([candidate_qg_input, pose_input])
        n_dim = concat_input.shape[2]._value
        n_filters = 32
        H = Conv2D(filters=n_filters,
                   kernel_size=(1, n_dim),
                   strides=(1, 1),
                   activation='relu',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(concat_input)
        for _ in range(2):
            H = Conv2D(filters=32,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       activation='relu',
                       kernel_initializer=self.kernel_initializer,
                       bias_initializer=self.bias_initializer)(H)
        q0_qg_eval = Conv2D(filters=1,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            activation='linear',
                            kernel_initializer=self.kernel_initializer,
                            bias_initializer=self.bias_initializer,
                            name='q0_qg_eval')(H)
        q0_qg_eval = Reshape((615,))(q0_qg_eval)
        #q0_qg_eval = RepeatVector(2)(q0_qg_eval)
        #q0_qg_eval = Reshape((615, 2, 1))(q0_qg_eval)
        return q0_qg_eval

    def construct_eval_net(self, candidate_qg_input):
        q0_qg_eval = self.construct_candidate_qg_from_q0_eval(candidate_qg_input)
        collision_input = Flatten()(self.collision_input)
        collision_input = RepeatVector(615)(collision_input)
        collision_input = Reshape((615, 615, 2))(collision_input)
        n_filters = 32
        n_dim = collision_input.shape[2]._value
        H = Conv2D(filters=n_filters,
                   kernel_size=(1, n_dim),
                   strides=(1, 1),
                   activation='relu',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(collision_input)
        for _ in range(2):
            H = Conv2D(filters=32,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       activation='relu',
                       kernel_initializer=self.kernel_initializer,
                       bias_initializer=self.bias_initializer)(H)
        q0_qg_eval = Conv2D(filters=1,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            activation='linear',
                            kernel_initializer=self.kernel_initializer,
                            bias_initializer=self.bias_initializer,
                            name='q0_qg_eval')(H)
        q0_qg_eval = Reshape((615,))(q0_qg_eval)
        evalnet = q0_qg_eval

        def compute_softmax(x):
            return K.softmax(x, axis=-1)
        import pdb;pdb.set_trace()

        evalnet = Lambda(compute_softmax, name='softmax')(evalnet)

        return evalnet
