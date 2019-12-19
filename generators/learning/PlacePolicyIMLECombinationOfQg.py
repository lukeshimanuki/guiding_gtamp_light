from keras.layers import *
from keras import backend as K
from keras.models import Model

from PlacePolicyIMLE import PlacePolicyIMLE


class PlacePolicyIMLECombinationOfQg(PlacePolicyIMLE):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        PlacePolicyIMLE.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.weight_file_name = 'place_imle_qg_combination_with_qg_evalnet_seed_%d' % config.seed

    def construct_policy_output(self):
        candidate_qg = self.construct_value_output()
        evalnet_input = Reshape((self.n_key_confs, 4, 1))(candidate_qg)
        eval_net = self.construct_eval_net(evalnet_input)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]), name='policy_output')([eval_net, candidate_qg])
        return output

    def construct_model(self, output, name):
        model = Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input, self.noise_input],
                      outputs=[output],
                      name=name)
        return model

    def construct_eval_net(self, qg_candidates):
        pose_input = RepeatVector(self.n_key_confs)(self.pose_input)
        pose_input = Reshape((self.n_key_confs, self.dim_poses, 1))(pose_input)

        collision_inp = Flatten()(self.collision_input)
        collision_inp = RepeatVector(self.n_key_confs)(collision_inp)
        collision_inp = Reshape((self.n_key_confs, self.n_key_confs * 2, 1))(collision_inp)
        concat_input = Concatenate(axis=2)([pose_input, qg_candidates, collision_inp])
        n_dim = concat_input.shape[2]._value
        dense_num = 32
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
        H = Conv2D(filters=1,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   activation='linear',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(H)
        H = Reshape((self.n_key_confs,))(H)

        def compute_softmax(x):
            return K.softmax(x, axis=-1)

        evalnet = Lambda(compute_softmax, name='softmax')(H)
        self.evalnet_model = self.construct_model(evalnet, 'evalnet_model')

        return evalnet

    def construct_value_output(self):
        pose_input = RepeatVector(self.n_key_confs)(self.pose_input)
        pose_input = Reshape((self.n_key_confs, self.dim_poses, 1))(pose_input)

        noise_input = RepeatVector(self.n_key_confs)(self.noise_input)
        noise_input = Reshape((self.n_key_confs, self.dim_noise, 1))(noise_input)
        concat_input = Concatenate(axis=2)([pose_input, noise_input, self.key_config_input])

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
        self.value_model = self.construct_model(value, 'valuemodel')
        return value
