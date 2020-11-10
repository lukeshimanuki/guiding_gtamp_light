from torch import nn

import torch
import os


class FullyConnected(nn.Module):
    def __init__(self, config):
        self.dim_input = 27
        self.dim_actions = (8, 2)
        self.n_hidden = 32
        self.config = config

        nn.Module.__init__(self)
        self.pose_net = \
            nn.Sequential(
                torch.nn.Linear(self.dim_input, self.n_hidden),
                nn.ReLU(),
                torch.nn.Linear(self.n_hidden, self.n_hidden),
                nn.ReLU(),
                torch.nn.Linear(self.n_hidden, 11*2))

        self.weight_dir,  self.weight_file_name = self.create_weight_file_name()
        if not os.path.isdir(self.weight_dir):
            os.makedirs(self.weight_dir)

        # todo try something that is invariant to the object poses?
        #   perhaps we could try to process each pose, and then aggregate using averaging?

    def create_weight_file_name(self):
        filedir = './learn/pose_based_q_function_weights/'
        filename = "weight_"
        filename += '_'.join(arg + "_" + str(getattr(self.config, arg)) for arg in [
            'optimizer',
            'seed',
            'lr',
            'operator',
            'n_hidden',
            'top_k',
            'num_train',
            'use_region_agnostic',
            'loss',
        ])
        filename += '.hdf5'
        print "Config:", filename
        return filedir, filedir + filename

    def forward(self, poses):
        pose_val = self.pose_net(poses)
        pose_val = pose_val.reshape((len(pose_val), 11, 2))
        return pose_val

    def load_weights(self):
        raise NotImplementedError
