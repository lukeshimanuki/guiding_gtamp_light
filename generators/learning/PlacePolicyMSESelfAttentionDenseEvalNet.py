from keras.layers import *
from keras.layers.merge import Concatenate
from generators.learning.PlacePolicyMSE import PlacePolicyMSE
from keras.models import Model
from keras import backend as K

import socket

if socket.gethostname() == 'lab' or socket.gethostname() == 'phaedra':
    ROOTDIR = './'
else:
    ROOTDIR = '/data/public/rw/pass.port/guiding_gtamp/'


def G_loss(true_actions, pred):
    return -K.mean(pred, axis=-1)


def slice_x(x):
    return x[:, 0:1]


def slice_y(x):
    return x[:, 1:2]


def slice_th(x):
    return x[:, 2:]


class PlacePolicyMSESelfAttentionDenseEvalNet(PlacePolicyMSE):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        PlacePolicyMSE.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.weight_file_name = 'place_mse_selfattention_seed_%d' % config.seed

    def construct_policy_output(self):
        # generating candidate q_g
        tiled_pose = self.get_tiled_input(self.pose_input)
        qk_goalflags_input = Concatenate(axis=2)(
            [self.key_config_input, self.goal_flag_input])
        candidate_qg = self.construct_value_output(qk_goalflags_input)

        candidate_qg = Reshape((615,4,1))(candidate_qg)
        evalnet_input = Concatenate(axis=2)([candidate_qg, self.goal_flag_input])
        eval_net = self.construct_eval_net(evalnet_input)
        candidate_qg = Reshape((615,4))(candidate_qg)

        output = Lambda(lambda x: K.batch_dot(x[0], x[1]), name='policy_output')([eval_net, candidate_qg])
        return output

    def construct_value_output(self, concat_input):
        # Computes the candidate goal configurations
        # q_g = phi_2(x_i), for some x_i
        dim_value_input = concat_input.shape[2]._value
        value = self.create_conv_layers(concat_input, dim_value_input, n_filters=128, use_pooling=False, use_flatten=False)
        value = Conv2D(filters=4,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       activation='linear',
                       kernel_initializer=self.kernel_initializer,
                       bias_initializer=self.bias_initializer,
                       name='value_output')(value)

        value = Lambda(lambda x: K.squeeze(x, axis=2), name='key_config_transformation')(value)
        self.value_model = Model(
            inputs=[self.goal_flag_input, self.key_config_input, self.pose_input],
            outputs=value,
            name='value_model')
        return value

    def construct_eval_net(self, candidate_qg_goal_flag_input):
        # Computes how important each candidate q_g is.
        # w_i = phi_1(x_i)
        # It currently takes in candidate q_g as an input

        # There currently are 615 candidate goal configurations
        concat_input = Concatenate(axis=2)([candidate_qg_goal_flag_input, self.collision_input])
        dim_input = concat_input.shape[2]._value

        collision_input = Flatten()(self.collision_input)
        dense_num = 32
        evalnet = Dense(dense_num, activation='relu',
                              kernel_initializer=self.kernel_initializer,
                              bias_initializer=self.bias_initializer)(collision_input)
        evalnet = Dense(615, activation='linear',
                              kernel_initializer=self.kernel_initializer,
                              bias_initializer=self.bias_initializer)(evalnet)

        def compute_softmax(x):
            #x = K.squeeze(x, axis=-1)
            #x = K.squeeze(x, axis=-1)
            return K.softmax(x, axis=-1)

        evalnet = Lambda(compute_softmax, name='softmax')(evalnet)

        return evalnet

    def construct_policy_model(self):
        mse_model = Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input],
                          outputs=self.policy_output,
                          name='policy_output')
        mse_model.compile(loss='mse', optimizer=self.opt_D)
        return mse_model

    def compute_policy_mse(self, data):
        pred = self.policy_model.predict(
            [data['goal_flags'], data['rel_konfs'], data['states'], data['poses']])
        return np.mean(np.power(pred - data['actions'], 2))
