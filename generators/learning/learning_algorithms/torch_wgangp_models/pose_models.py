import torch
from torch import nn
from models import BaseModel


class Discriminator(BaseModel):
    def __init__(self, dim_data, atype, region, problem_name):
        BaseModel.__init__(self, atype, region, problem_name)

        dim_all_object_and_robot_poses = 27
        self.pose_net = \
            nn.Sequential(
                torch.nn.Linear(dim_all_object_and_robot_poses, self.n_hidden),
                nn.ReLU(),
                torch.nn.Linear(self.n_hidden, self.n_hidden),
                nn.ReLU()
            )

        dim_actions = dim_data
        self.action_net = \
            nn.Sequential(
                torch.nn.Linear(dim_actions, self.n_hidden),
                nn.ReLU(),
                torch.nn.Linear(self.n_hidden, self.n_hidden),
                nn.ReLU()
            )

        dim_input = self.n_hidden * 2
        self.output = \
            nn.Sequential(
                torch.nn.Linear(dim_input, self.n_hidden),
                nn.ReLU(),
                torch.nn.Linear(self.n_hidden, 1)
            )

    def forward(self, action, _, poses):
        pose_val = self.pose_net(poses)
        action_val = self.action_net(action)
        concat = torch.cat((pose_val, action_val), -1)
        return self.output(concat)


class Generator(BaseModel):
    def __init__(self, dim_data, atype, region, problem_name):
        BaseModel.__init__(self, atype, region, problem_name)

        dim_all_object_and_robot_poses_and_noise = 30
        self.pose_net = \
            nn.Sequential(
                torch.nn.Linear(dim_all_object_and_robot_poses_and_noise, self.n_hidden),
                nn.ReLU(),
                torch.nn.Linear(self.n_hidden, self.n_hidden),
                nn.ReLU()
            )

        dim_actions = dim_data
        dim_input = self.n_hidden + dim_actions
        self.output = \
            nn.Sequential(
                torch.nn.Linear(dim_input, self.n_hidden),
                nn.ReLU(),
                torch.nn.Linear(self.n_hidden, dim_actions)
            )

    def forward(self, konf, poses, noise):
        pose_val = self.pose_net(poses)
        concat = torch.cat((pose_val, noise), -1)
        return self.output(concat)
