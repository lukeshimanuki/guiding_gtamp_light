from keras.layers import *
from keras.layers.merge import Concatenate
from keras.models import Model

from PlacePolicyMSE import PlacePolicyMSE


class PlacePolicyMSEFeedForward(PlacePolicyMSE):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        PlacePolicyMSE.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.weight_file_name = 'place_mse_ff_seed_%d' % config.seed

    def construct_policy_output(self):
        tiled_pose = self.get_tiled_input(self.pose_input)
        concat_input = Concatenate(axis=2)(
            [self.key_config_input, self.goal_flag_input, self.collision_input, tiled_pose])
        H = Flatten()(concat_input)
        dense_num = 256
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
        mse_model = Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input],
                          outputs=self.policy_output,
                          name='policy_output')
        mse_model.compile(loss='mse', optimizer=self.opt_D)
        return mse_model


