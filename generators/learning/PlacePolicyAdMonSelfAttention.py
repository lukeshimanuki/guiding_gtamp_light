from PlacePolicyAdMon import PlacePolicyAdMon
from keras.models import Model
from keras.layers import *
from keras import backend as K


class PlacePolicyAdMonSelfAttention(PlacePolicyAdMon):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        PlacePolicyAdMon.__init__(self, dim_action, dim_collision, save_folder, tau, config)

    def construct_critic(self):
        collision_input = Flatten()(self.collision_input)
        concat_input = Concatenate(axis=1, name='q0_qg_ck')([self.pose_input, self.action_input, collision_input])
        eval_net = self.construct_eval_net(concat_input)
        key_config_input = Reshape((self.n_key_confs, 4))(self.key_config_input)
        best_qk = Lambda(lambda x: K.batch_dot(x[0], x[1]), name='best_qk')([eval_net, key_config_input])
        output = self.construct_critic_value_based_on_qk(best_qk)
        return output

    def construct_policy_output(self):
        collision_input = Flatten()(self.collision_input)
        concat_input = Concatenate(axis=1, name='q0_ck')([self.pose_input, collision_input])
        eval_net = self.construct_eval_net(concat_input)
        key_config_input = Reshape((self.n_key_confs, 4))(self.key_config_input)
        best_qk = Lambda(lambda x: K.batch_dot(x[0], x[1]), name='best_qk')([eval_net, key_config_input])
        self.best_qk_model = self.construct_model(best_qk, 'best_qk')
        output = self.construct_policy_output_based_on_best_qk(best_qk)
        return output

    def construct_eval_net(self, concat_input):
        evalnet = Dense(64, activation='relu',
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer)(concat_input)
        evalnet = Dense(32, activation='relu',
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer)(evalnet)
        evalnet = Dense(self.n_key_confs, activation='linear',
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer, name='collision_feature')(evalnet)
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
        return evalnet

    def construct_policy_output_based_on_best_qk(self, best_qk):
        concat = Concatenate(axis=-1)([self.pose_input, best_qk, self.noise_input])
        value = Dense(32, activation='linear',
                      kernel_initializer=self.kernel_initializer,
                      bias_initializer=self.bias_initializer)(concat)
        value = Dense(4, activation='linear',
                      kernel_initializer=self.kernel_initializer,
                      bias_initializer=self.bias_initializer, name='policy_output')(value)
        return value

    def construct_critic_value_based_on_qk(self, best_qk):
        concat = Concatenate(axis=-1)([self.pose_input, best_qk])
        value = Dense(32, activation='relu',
                      kernel_initializer=self.kernel_initializer,
                      bias_initializer=self.bias_initializer)(concat)
        value = Dense(1, activation='linear',
                      kernel_initializer=self.kernel_initializer,
                      bias_initializer=self.bias_initializer, name='critic_output')(value)
        return value
