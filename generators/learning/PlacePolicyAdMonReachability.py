from PlacePolicyAdMonSelfAttention import PlacePolicyAdMonSelfAttention
from keras.layers import *
from keras.models import Model

import tensorflow as tf


class PlacePolicyAdMonReachability(PlacePolicyAdMonSelfAttention):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        PlacePolicyAdMonSelfAttention.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.weight_file_name = 'place_sa_admon_seed_%d' % config.seed

    def construct_policy_loss_model(self):
        for layer in self.critic_model.layers:
            # for some obscure reason, disc weights still get updated when self.disc.fit is called
            # I speculate that this has to do with the status of the layers at the time it was compiled
            layer.trainable = False

        DG_output = self.critic_model([self.policy_output, self.goal_flag_input, self.key_config_input,
                                       self.collision_input, self.pose_input])
        model = Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input,
                              self.noise_input], outputs=[DG_output])
        model.compile(loss=lambda _, x: -tf.reduce_mean(x),
                      optimizer=self.opt_G,
                      metrics=[])

        return model

    def construct_critic(self):
        # todo build in the reachability reasoning in here
        a_input = RepeatVector(self.n_key_confs)(self.action_input)
        p_input = RepeatVector(self.n_key_confs)(self.pose_input)
        a_input = Reshape((self.n_key_confs, self.dim_action, 1))(a_input)
        p_input = Reshape((self.n_key_confs, self.dim_poses, 1))(p_input)
        critic_input = Concatenate(axis=2, name='q0_qg_ck')([a_input, p_input, self.collision_input])

        dense_num = 32
        n_dim = critic_input.shape[2]._value

        H = Conv2D(filters=dense_num,
                   kernel_size=(1, n_dim),
                   strides=(1, 1),
                   activation='relu',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(critic_input)
        H = Conv2D(filters=dense_num,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   activation='relu',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(H)
        H = Conv2D(filters=dense_num,
                   kernel_size=(4, 1),
                   strides=(2, 1),
                   activation='relu',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(H)
        H = Conv2D(filters=dense_num,
                   kernel_size=(4, 1),
                   strides=(2, 1),
                   activation='relu',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(H)
        H = Conv2D(filters=dense_num,
                   kernel_size=(4, 1),
                   strides=(2, 1),
                   activation='relu',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(H)
        H = MaxPooling2D(pool_size=(4, 1))(H)
        H = Flatten()(H)
        critic = Dense(32, activation='relu',
                       kernel_initializer=self.kernel_initializer,
                       bias_initializer=self.bias_initializer)(H)
        critic = Dense(1, activation='linear',
                       kernel_initializer=self.kernel_initializer,
                       bias_initializer=self.bias_initializer)(critic)
        return critic
