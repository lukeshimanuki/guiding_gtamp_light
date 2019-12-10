from keras.layers import *
from keras import backend as K
from keras.models import Model

from PlacePolicyIMLE import PlacePolicyIMLE


class PlacePolicyIMLESelfAttention(PlacePolicyIMLE):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        PlacePolicyIMLE.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.weight_file_name = 'place_imle_ff_seed_%d' % config.seed

    def construct_policy_output(self):
        candidate_qg = self.construct_qg_candidate_generator()
        evalnet_input = Reshape((615, 4, 1))(candidate_qg)
        eval_net = self.construct_eval_net(evalnet_input)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]), name='policy_output')([eval_net, candidate_qg])
        return output

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
            return K.softmax(x, axis=-1)

        #q0_qg_eval = Lambda(compute_softmax, name='softmax_q0_qg')(q0_qg_eval)
        #q0_qg_eval = Reshape((615,))(q0_qg_eval)
        return q0_qg_eval

    def construct_eval_net(self, candidate_qg_input):
        q0_qg_eval = self.construct_q0_qg_eval(candidate_qg_input)
        collision_input = Flatten()(self.collision_input)

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

    def construct_qg_candidate_generator(self):
        # todo take the noise into account
        noise_input = RepeatVector(615)(self.noise_input)
        noise_input = Reshape((615, self.dim_noise, 1))(noise_input)
        concat_input = Concatenate(axis=2)([self.key_config_input, noise_input])
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
                       bias_initializer=self.bias_initializer,
                       name='value_output')(H)

        value = Lambda(lambda x: K.squeeze(x, axis=2), name='key_config_transformation')(value)
        self.value_model = Model(
            inputs=[self.goal_flag_input, self.key_config_input, self.pose_input, self.noise_input],
            outputs=value,
            name='value_model')
        return value



