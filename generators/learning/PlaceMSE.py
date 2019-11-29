from keras.layers import *
from keras.layers.merge import Concatenate
from generators.learning.Policy import Policy
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


class PlaceMSE(Policy):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        Policy.__init__(self, dim_action, dim_collision, save_folder, tau)

        self.dim_poses = 8
        self.dim_collision = dim_collision

        # setup inputs
        self.action_input = Input(shape=(dim_action,), name='a', dtype='float32')  # action
        self.collision_input = Input(shape=dim_collision, name='s', dtype='float32')  # collision vector
        self.pose_input = Input(shape=(self.dim_poses,), name='pose', dtype='float32')  # pose
        self.key_config_input = Input(shape=(615, 4, 1), name='konf', dtype='float32')  # relative key config
        self.goal_flag_input = Input(shape=(615, 4, 1), name='goal_flag',
                                     dtype='float32')  # goal flag (is_goal_r, is_goal_obj)

        # setup inputs related to detecting whether a key config is relevant
        self.cg_input = Input(shape=(dim_action,), name='cg', dtype='float32')  # action
        self.ck_input = Input(shape=(dim_action,), name='ck', dtype='float32')  # action
        self.collision_at_each_ck = Input(shape=(2,), name='ck', dtype='float32')  # action

        self.weight_file_name = 'admonpose_seed_%d' % config.seed
        self.pretraining_file_name = 'pretrained_%d.h5' % config.seed
        self.seed = config.seed

        self.policy_output = self.construct_self_attention_policy_output()
        self.policy_model = self.construct_policy_model()

    def construct_self_attention_policy_output(self):
        tiled_pose = self.get_tiled_input(self.pose_input)
        concat_input = Concatenate(axis=2)(
            [self.key_config_input, self.goal_flag_input, self.collision_input, tiled_pose])
        dim_input = concat_input.shape[2]._value

        # The query matrix
        query = self.create_conv_layers(concat_input, dim_input, use_pooling=False, use_flatten=False)
        query = Conv2D(filters=1,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       activation='linear',
                       kernel_initializer=self.kernel_initializer,
                       bias_initializer=self.bias_initializer)(query)

        def compute_W(x):
            x = K.squeeze(x, axis=-1)
            x = K.squeeze(x, axis=-1)
            return K.softmax(x*100, axis=-1)

        W = Lambda(compute_W, name='softmax')(query)

        self.w_model = Model(
            inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input],
            outputs=W,
            name='w_model')

        # The value matrix
        value = self.create_conv_layers(concat_input, dim_input, use_pooling=False, use_flatten=False)
        value = Conv2D(filters=4,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       activation='linear',
                       kernel_initializer=self.kernel_initializer,
                       bias_initializer=self.bias_initializer,
                       )(value)

        value = Lambda(lambda x: K.squeeze(x, axis=2), name='key_config_transformation')(value)
        self.value_model = Model(
            inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input],
            outputs=value,
            name='value_model')
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([W, value])
        return output

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

    def train_policy(self, states, poses, rel_konfs, goal_flags, actions, sum_rewards, epochs=500):
        train_idxs, test_idxs = self.get_train_and_test_indices(len(actions))
        train_data, test_data = self.get_train_and_test_data(states, poses, rel_konfs, goal_flags,
                                                             actions, sum_rewards,
                                                             train_idxs, test_idxs)
        callbacks = self.create_callbacks_for_pretraining()

        actions = train_data['actions']
        goal_flags = train_data['goal_flags']
        poses = train_data['poses']
        rel_konfs = train_data['rel_konfs']
        collisions = train_data['states']
        pre_mse = self.compute_policy_mse(test_data)
        self.policy_model.fit([goal_flags, rel_konfs, collisions, poses], actions,
                              batch_size=32,
                              epochs=epochs,
                              verbose=2,
                              callbacks=callbacks,
                              validation_split=0.1, shuffle=False)
        post_mse = self.compute_policy_mse(test_data)
        print "Pre-and-post test errors", pre_mse, post_mse
        # wvals = self.W_model.predict([goal_flags, rel_konfs, collisions, poses])[0]
        collision_idxs = collisions[0].squeeze()[:, 0] == True

