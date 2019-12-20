from keras.optimizers import *
from keras.layers import *
from keras.callbacks import *
from keras import initializers

import os
import numpy as np
import pickle
import socket

if socket.gethostname() == 'lab' or socket.gethostname() == 'phaedra' or socket.gethostname() == 'dell-XPS-15-9560':
    ROOTDIR = './'
else:
    ROOTDIR = '/data/public/rw/pass.port/guiding_gtamp_light/learned_weights/'


# Implements util functions and initializes dimension variables and directories.
class Policy:
    def __init__(self, dim_action, dim_state, save_folder, tau):
        self.dim_noise = 4
        self.noise_input = Input(shape=(self.dim_noise,), name='noise_input', dtype='float32')

        if save_folder !='' and not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        self.opt_G = Adam(lr=1e-4, beta_1=0.5)
        self.opt_D = Adam(lr=1e-3, beta_1=0.5)

        # initialize
        self.kernel_initializer = initializers.glorot_uniform()
        self.bias_initializer = initializers.glorot_uniform()

        if dim_action < 10:
            dim_z = dim_action
        else:
            dim_z = int(dim_action / 2)

        # setup dimensions for inputs
        self.dim_action = dim_action
        self.dim_state = dim_state
        self.n_key_confs = dim_state[0]

        # setup inputs
        """
        self.tau = tau
        self.tau_input = Input(shape=(1,), name='tau', dtype='float32')  # collision vector
        """
        self.save_folder = save_folder

        self.test_data = None
        self.desired_test_err = None
        self.disc = None
        self.disc_mse_model = None
        self.weight_file_name = None
        self.seed = None

    @staticmethod
    def get_batch_size(n_data):
        batch_size = np.min([32, n_data])
        if batch_size == 0:
            batch_size = 1
        return batch_size

    def get_tiled_input(self, inp):
        inp_dim = inp.shape[1]._value
        repeated = RepeatVector(self.n_key_confs)(inp)
        repeated = Reshape((self.n_key_confs, inp_dim, 1))(repeated)
        return repeated

    def set_learning_rates(self, d_lr, g_lr):
        K.set_value(self.opt_G.lr, g_lr)
        K.set_value(self.opt_D.lr, d_lr)

    def get_train_and_test_indices(self, n_data):
        test_idxs = np.random.randint(0, n_data, size=int(0.2 * n_data))
        train_idxs = list(set(range(n_data)).difference(set(test_idxs)))
        #pickle.dump({'train': train_idxs, 'test': test_idxs},
        #            open('data_idxs_seed_%s' % self.seed, 'wb'))
        return train_idxs, test_idxs

    @staticmethod
    def get_train_and_test_indices_based_on_sum_rewards(n_data, sum_rewards):
        probability_of_being_sampled = (np.exp(sum_rewards) / np.sum(np.exp(sum_rewards))).squeeze()
        indices_based_on_sum_rewards = np.random.choice(n_data, n_data, p=probability_of_being_sampled)

        test_idxs = np.random.randint(0, n_data, size=int(0.4 * n_data))
        train_idxs = list(set(range(n_data)).difference(set(test_idxs)))

        test_idxs = indices_based_on_sum_rewards[test_idxs]
        train_idxs = indices_based_on_sum_rewards[train_idxs]
        test_idxs = np.array([idx for idx in test_idxs if idx not in train_idxs])
        return train_idxs, test_idxs

    def save_weights(self, additional_name=''):
        fdir = ROOTDIR + '/' + self.save_folder + '/'
        fname = self.weight_file_name + additional_name + '.h5'
        if not os.path.isdir(fdir):
            os.makedirs(fdir)
        self.policy_model.save_weights(fdir + fname)

    def load_weights(self, additional_name=''):
        fdir = ROOTDIR + '/' + self.save_folder + '/'
        fname = self.weight_file_name + additional_name + '.h5'
        print "Loading weights", fdir+fname
        self.policy_model.load_weights(fdir+fname)

    def load_best_weights(self):
        print "Loading weights", self.save_folder + self.weight_file_name + 'best_val_err.h5'
        self.policy_model.load_weights(self.save_folder + self.weight_file_name + 'best_val_err.h5')

    @staticmethod
    def get_train_and_test_data(states, konf_relevance, poses, rel_konfs, goal_flags, actions, sum_rewards,
                                train_indices,
                                test_indices):
        train = {'states': states[train_indices, :],
                 'poses': poses[train_indices, :],
                 'actions': actions[train_indices, :],
                 'rel_konfs': rel_konfs[train_indices, :],
                 'sum_rewards': sum_rewards[train_indices, :],
                 'goal_flags': goal_flags[train_indices, :],
                 'konf_relevance': konf_relevance[train_indices, :]
                 }
        test = {'states': states[test_indices, :],
                'poses': poses[test_indices, :],
                'goal_flags': goal_flags[test_indices, :],
                'actions': actions[test_indices, :],
                'rel_konfs': rel_konfs[test_indices, :],
                'sum_rewards': sum_rewards[test_indices, :],
                'konf_relevance': konf_relevance[test_indices, :]
                }
        return train, test

    @staticmethod
    def get_batch(cols, goal_flags, poses, rel_konfs, actions, sum_rewards, batch_size):
        n_data = len(cols)
        indices = np.random.permutation(range(n_data))[0:batch_size]
        cols_batch = np.array(cols[indices, :])  # collision vector
        goal_flag_batch = np.array(goal_flags[indices, :])  # collision vector
        a_batch = np.array(actions[indices, :])
        pose_batch = np.array(poses[indices, :])
        konf_batch = np.array(rel_konfs[indices, :])
        sum_reward_batch = np.array(sum_rewards[indices, :])
        return cols_batch, goal_flag_batch, pose_batch, konf_batch, a_batch, sum_reward_batch

    def create_conv_layers(self, input, n_filters=32, use_pooling=True, use_flatten=True):
        # a helper function for creating a NN that applies the same function for each key config
        n_dim = input.shape[2]._value
        H = Conv2D(filters=n_filters,
                   kernel_size=(1, n_dim),
                   strides=(1, 1),
                   activation='linear',
                   # why does this have to be linear again? For predicting a value that will be soft-maxed, the values tend to saturate.
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(input)
        H = LeakyReLU()(H)
        for _ in range(2):
            H = Conv2D(filters=n_filters,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       activation='linear',
                       kernel_initializer=self.kernel_initializer,
                       bias_initializer=self.bias_initializer)(H)
            H = LeakyReLU()(H)
        if use_pooling:
            H = MaxPooling2D(pool_size=(2, 1))(H)
        if use_flatten:
            H = Flatten()(H)
        return H
