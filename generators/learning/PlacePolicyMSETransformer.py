from keras.layers import *
from keras.layers.merge import Concatenate
from generators.learning.PlacePolicyMSE import PlacePolicyMSE
from keras.models import Model
from keras import backend as K

import socket
import tensorflow as tf
if socket.gethostname() == 'lab' or socket.gethostname() == 'phaedra':
    ROOTDIR = './'
else:
    ROOTDIR = '/data/public/rw/pass.port/guiding_gtamp/'


class PlacePolicyMSETransformer(PlacePolicyMSE):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        PlacePolicyMSE.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.weight_file_name = 'place_sa_mse_seed_%d' % config.seed

    def construct_policy_model(self):
        # noise input is used to make the prediction format consistent with imle
        mse_model = Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input,
                                  self.noise_input],
                          outputs=self.policy_output,
                          name='policy_output')
        mse_model.compile(loss='mse', optimizer=self.opt_D)
        return mse_model

    def construct_transformer_block(self, input, output_dim):
        n_dim = input.shape[2]._value
        n_filters = 32
        H = Conv2D(filters=n_filters,
                   kernel_size=(1, n_dim),
                   strides=(1, 1),
                   activation='relu',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(input)
        H = MaxPooling2D(pool_size=(3, 1))(H)
        H = Conv2D(filters=n_filters,
                   kernel_size=(3, 1),
                   strides=(3, 1),
                   activation='relu',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(H)
        H = MaxPooling2D(pool_size=(3, 1))(H)
        H = Conv2D(filters=output_dim,
                   kernel_size=(3, 1),
                   strides=(3, 1),
                   activation='relu',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(H)
        H = Flatten()(H)
        value = Dense(self.n_key_confs*output_dim, activation='linear',
                      kernel_initializer=self.kernel_initializer,
                      bias_initializer=self.bias_initializer)(H)
        value = Reshape((self.n_key_confs, output_dim, ))(value)
        return value

    def construct_policy_output(self):
        pose_input = RepeatVector(self.n_key_confs)(self.pose_input)
        pose_input = Reshape((self.n_key_confs, self.dim_poses, 1))(pose_input)
        input = Concatenate(axis=2)([pose_input, self.key_config_input, self.collision_input])

        query_matrix = self.construct_transformer_block(input, output_dim=16)  # n by d
        key_matrix = self.construct_transformer_block(input, output_dim=16)  # n by d
        value_matrix = self.construct_transformer_block(input, output_dim=4)  # n by d
        weights = Dot(axes=-1)([query_matrix, key_matrix])
        self.key_model = self.construct_model(key_matrix, 'key')
        self.query_model = self.construct_model(query_matrix, 'query')
        self.weight_model = self.construct_model(weights, 'weights')

        def add_wrt_last_axis(x):
            return tf.reduce_sum(x, axis=-1)

        value_weights = Lambda(add_wrt_last_axis)(weights)

        def compute_softmax(x):
            return K.softmax(x, axis=-1)

        weights = Lambda(compute_softmax, name='softmax')(value_weights)  # n by d, with softmax

        # weights = \sum_{i} \sum_{j}q_i k_j
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]), name='output')([weights, value_matrix])
        return output
