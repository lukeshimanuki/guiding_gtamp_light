from keras.layers import *
from keras.layers.merge import Concatenate
from keras.models import Model

from PlacePolicyIMLE import PlacePolicyIMLE


class PlacePolicyIMLEFeedForward(PlacePolicyIMLE):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        PlacePolicyIMLE.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.weight_file_name = 'place_mse_ff_seed_%d' % config.seed

    def construct_policy_output(self):
        self.noise_input = Input(shape=(self.dim_noise,), name='z', dtype='float32')
        tiled_pose = self.get_tiled_input(self.pose_input)
        tiled_noise = self.get_tiled_input(self.noise_input)
        concat_input = Concatenate(axis=2)(
            [self.key_config_input, self.collision_input, tiled_noise])
        H = Flatten()(concat_input)
        dense_num = 32
        hidden_action = Dense(dense_num, activation='relu',
                              kernel_initializer=self.kernel_initializer,
                              bias_initializer=self.bias_initializer)(H)
        hidden_action = Dense(dense_num, activation='relu',
                              kernel_initializer=self.kernel_initializer,
                              bias_initializer=self.bias_initializer)(hidden_action)
        action_output = Dense(self.dim_action,
                              activation='linear',
                              kernel_initializer=self.kernel_initializer,
                              bias_initializer=self.bias_initializer,
                              name='policy_output')(hidden_action)

        return action_output

    def construct_policy_model(self):
        model = Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input,
                              self.noise_input],
                      outputs=[self.policy_output],
                      name='policy_model')
        model.compile(loss='mse', optimizer=self.opt_D)
        return model

