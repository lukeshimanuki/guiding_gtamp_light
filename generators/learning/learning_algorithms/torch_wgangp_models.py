import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, dim_data):
        nn.Module.__init__(self)
        n_hidden = 32
        n_konfs = 618 * 2
        self.konf_net = \
            nn.Sequential(
                torch.nn.Linear(n_konfs, n_hidden),
                nn.ReLU(),
                torch.nn.Linear(n_hidden, n_hidden),
                nn.ReLU()
            )

        dim_poses = 24
        self.pose_net = \
            nn.Sequential(
                torch.nn.Linear(dim_poses, n_hidden),
                nn.ReLU(),
                torch.nn.Linear(n_hidden, n_hidden),
                nn.ReLU()
            )

        dim_actions = dim_data
        self.action_net = \
            nn.Sequential(
                torch.nn.Linear(dim_actions, n_hidden),
                nn.ReLU(),
                torch.nn.Linear(n_hidden, n_hidden),
                nn.ReLU()
            )

        dim_input = n_hidden * 3
        self.output = \
            nn.Sequential(
                torch.nn.Linear(dim_input, n_hidden),
                nn.ReLU(),
                torch.nn.Linear(n_hidden, 1)
            )

    def forward(self, action, konf, pose):
        konf = konf.view((-1, 618 * 2))
        konf_val = self.konf_net(konf)
        pose_val = self.pose_net(pose)
        action_val = self.action_net(action)
        concat = torch.cat((konf_val, pose_val, action_val), -1)
        return self.output(concat)


class Generator(nn.Module):
    def __init__(self, dim_data):
        nn.Module.__init__(self)
        n_hidden = 32
        n_konfs = 618 * 2
        self.konf_net = \
            nn.Sequential(
                torch.nn.Linear(n_konfs, n_hidden),
                nn.ReLU(),
                torch.nn.Linear(n_hidden, n_hidden),
                nn.ReLU()
            )

        dim_poses = 24
        self.pose_net = \
            nn.Sequential(
                torch.nn.Linear(dim_poses, n_hidden),
                nn.ReLU(),
                torch.nn.Linear(n_hidden, n_hidden),
                nn.ReLU()
            )

        dim_actions = dim_data
        dim_input = n_hidden * 2 + dim_actions
        self.output = \
            nn.Sequential(
                torch.nn.Linear(dim_input, n_hidden),
                nn.ReLU(),
                torch.nn.Linear(n_hidden, dim_actions)
            )

    def forward(self, konf, pose, noise):
        konf = konf.view((-1, 618 * 2))
        konf_val = self.konf_net(konf)
        pose_val = self.pose_net(pose)
        concat = torch.cat((konf_val, pose_val, noise), -1)

        return self.output(concat)
