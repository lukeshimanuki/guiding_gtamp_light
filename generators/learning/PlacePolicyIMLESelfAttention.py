from keras.layers import *
from keras import backend as K
from keras.models import Model

from PlacePolicyIMLE import PlacePolicyIMLE


class PlacePolicyIMLESelfAttention(PlacePolicyIMLE):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        PlacePolicyIMLE.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.weight_file_name = 'place_imle_ff_seed_%d' % config.seed

    def construct_policy_output(self):
        eval_net = self.construct_eval_net(0)
        key_config_input = Reshape((self.n_key_confs, 4))(self.key_config_input)
        best_qk = Lambda(lambda x: K.batch_dot(x[0], x[1]), name='best_qk')([eval_net, key_config_input])
        output = self.construct_qg_candidate_generator(best_qk)
        return output

    def construct_eval_net(self, candidate_qg_input):
        q_0 = self.pose_input
        q_0 = RepeatVector(self.n_key_confs)(q_0)
        q_0 = Reshape((self.n_key_confs, self.dim_poses, 1))(q_0)
        concat_input = Concatenate(axis=2, name='q0_qg_qk_ck')([q_0, self.key_config_input, self.collision_input])
        # concat_input = Concatenate(axis=2, name='q0_qg_qk_ck')([q_0, candidate_qg])
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

    def construct_qg_candidate_generator(self, best_qk):
        concat = Concatenate(axis=-1)([self.pose_input, best_qk, self.noise_input])
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
        return value




