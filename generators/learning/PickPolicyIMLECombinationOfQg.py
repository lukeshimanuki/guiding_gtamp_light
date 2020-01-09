from keras.layers import *
from keras import backend as K
from keras.models import Model

from PlacePolicyIMLECombinationOfQg import PlacePolicyIMLECombinationOfQg


class PickPolicyIMLECombinationOfQg(PlacePolicyIMLECombinationOfQg):
    def __init__(self, dim_action, dim_collision, dim_poses, n_key_configs, save_folder, config):
        PlacePolicyIMLECombinationOfQg.__init__(self, dim_action, dim_collision, dim_poses, n_key_configs, save_folder, config)

    def construct_value_output(self):
        pose_input = RepeatVector(self.n_key_confs)(self.pose_input)
        pose_input = Reshape((self.n_key_confs, self.dim_poses, 1))(pose_input)

        noise_input = RepeatVector(self.n_key_confs)(self.noise_input)
        noise_input = Reshape((self.n_key_confs, self.dim_noise, 1))(noise_input)
        concat_input = Concatenate(axis=2)([self.goal_flag_input, pose_input, noise_input, self.key_config_input])

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
        value = Conv2D(filters=self.dim_action,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       activation='linear',
                       kernel_initializer=self.kernel_initializer,
                       bias_initializer=self.bias_initializer)(H)

        value = Lambda(lambda x: K.squeeze(x, axis=2), name='candidate_qg')(value)
        self.value_model = self.construct_model(value, 'valuemodel')
        return value
