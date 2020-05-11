import torch
from torch import nn
from models import BaseGenerator, BaseDiscriminator


class Discriminator(BaseDiscriminator):
    def __init__(self, dim_konf, dim_data, atype):
        BaseDiscriminator.__init__(self, dim_konf, atype)
        n_hidden = 32
        n_konfs = 618 * dim_konf
        self.konf_net = \
            nn.Sequential(
                torch.nn.Linear(n_konfs, n_hidden),
                nn.ReLU(),
                torch.nn.Linear(n_hidden, n_hidden),
                nn.ReLU()
            )

        dim_pose_ids = 8*3 + 2
        self.pose_net = \
            nn.Sequential(
                torch.nn.Linear(dim_pose_ids, n_hidden),
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

    def forward(self, action, konf, pose_ids):
        konf = konf.view((-1, 618 * self.dim_konf))
        konf_val = self.konf_net(konf)

        #target_obj_pose = pose_ids[:, 0:4]
        #robot_curr_pose_and_id = pose_ids[:, -6:]
        #pose_ids = torch.cat([target_obj_pose, robot_curr_pose_and_id], -1)

        pose_val = self.pose_net(pose_ids)
        action_val = self.action_net(action)
        concat = torch.cat((konf_val, pose_val, action_val), -1)
        return self.output(concat)


class Generator(BaseGenerator):
    def __init__(self, dim_konf, dim_data, atype):
        BaseGenerator.__init__(self, dim_konf, atype)
        n_hidden = 32
        n_konfs = 618 * dim_konf
        self.konf_net = \
            nn.Sequential(
                torch.nn.Linear(n_konfs, n_hidden),
                nn.ReLU(),
                torch.nn.Linear(n_hidden, n_hidden),
                nn.ReLU()
            )

        dim_pose_ids = 8*3 + 2
        self.pose_net = \
            nn.Sequential(
                torch.nn.Linear(dim_pose_ids, n_hidden),
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

    def forward(self, konf, pose_ids, noise):
        konf = konf.view((-1, 618 * self.dim_konf))
        konf_val = self.konf_net(konf)

        #target_obj_pose = pose_ids[:, 0:4]
        #robot_curr_pose_and_id = pose_ids[:, -6:]
        #pose_ids = torch.cat([target_obj_pose, robot_curr_pose_and_id], -1)

        pose_val = self.pose_net(pose_ids)
        concat = torch.cat((konf_val, pose_val, noise), -1)
        return self.output(concat)
