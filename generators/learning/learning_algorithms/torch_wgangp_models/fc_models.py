import torch
from torch import nn
from models import BaseModel


class Discriminator(BaseModel):
    def __init__(self, dim_data, atype, region, problem_name):
        BaseModel.__init__(self, atype, region, problem_name)

        self.konf_net = \
            nn.Sequential(
                torch.nn.Linear(self.n_konfs*self.dim_konf, self.n_hidden),
                nn.ReLU(),
                torch.nn.Linear(self.n_hidden, self.n_hidden),
                nn.ReLU()
            )
        self.pose_net = \
            nn.Sequential(
                torch.nn.Linear(self.dim_pose_ids, self.n_hidden),
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

        dim_input = self.n_hidden * 3
        self.output = \
            nn.Sequential(
                torch.nn.Linear(dim_input, self.n_hidden),
                nn.ReLU(),
                torch.nn.Linear(self.n_hidden, 1)
            )

    def forward(self, action, konf, pose_ids):
        konf, pose_ids = self.filter_data_according_to_cases(konf, pose_ids)
        konf = konf.reshape((-1, self.n_konfs*self.dim_konf))
        konf_val = self.konf_net(konf)

        pose_val = self.pose_net(pose_ids)
        action_val = self.action_net(action)
        concat = torch.cat((konf_val, pose_val, action_val), -1)
        return self.output(concat)


class Generator(BaseModel):
    def __init__(self, dim_data, atype, region, problem_name):
        BaseModel.__init__(self, atype, region, problem_name)

        self.konf_net = \
            nn.Sequential(
                torch.nn.Linear(self.n_konfs*self.dim_konf, self.n_hidden),
                nn.ReLU(),
                torch.nn.Linear(self.n_hidden, self.n_hidden),
                nn.ReLU()
            )

        self.pose_net = \
            nn.Sequential(
                torch.nn.Linear(self.dim_pose_ids, self.n_hidden),
                nn.ReLU(),
                torch.nn.Linear(self.n_hidden, self.n_hidden),
                nn.ReLU()
            )

        dim_actions = dim_data
        dim_input = self.n_hidden * 2 + dim_actions
        self.output = \
            nn.Sequential(
                torch.nn.Linear(dim_input, self.n_hidden),
                nn.ReLU(),
                torch.nn.Linear(self.n_hidden, dim_actions)
            )

    def forward(self, konf, pose_ids, noise):
        konf, pose_ids = self.filter_data_according_to_cases(konf, pose_ids)
        konf = konf.reshape((-1, self.n_konfs*self.dim_konf))
        konf_val = self.konf_net(konf)

        pose_val = self.pose_net(pose_ids)
        concat = torch.cat((konf_val, pose_val, noise), -1)
        return self.output(concat)
