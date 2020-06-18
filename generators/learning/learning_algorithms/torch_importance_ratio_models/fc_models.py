import torch
from torch import nn
from generators.learning.learning_algorithms.torch_wgangp_models.models import BaseModel


class FCImportanceRatioEstimator(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config.atype, config.region, config.domain)

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
        if config.atype == 'pick':
            dim_action = 7
        else:
            dim_action = 4

        self.action_net = \
            nn.Sequential(
                torch.nn.Linear(dim_action, self.n_hidden),
                nn.ReLU(),
                torch.nn.Linear(self.n_hidden, self.n_hidden),
                nn.ReLU()
            )

        dim_input = self.n_hidden * 3
        self.output = \
            nn.Sequential(
                torch.nn.Linear(dim_input, self.n_hidden),
                nn.ReLU(),
                torch.nn.Linear(self.n_hidden, 1),
            )

    def forward(self, action, konf, pose_ids):
        konf, pose_ids = self.filter_data_according_to_cases(konf, pose_ids)
        konf = konf.reshape((-1, self.n_konfs*self.dim_konf))
        konf_val = self.konf_net(konf)

        pose_val = self.pose_net(pose_ids)
        action_val = self.action_net(action)
        concat = torch.cat((konf_val, pose_val, action_val), -1)
        return self.output(concat)
