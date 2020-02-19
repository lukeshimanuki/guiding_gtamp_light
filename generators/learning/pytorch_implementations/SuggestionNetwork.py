import torch
from torch import nn


class SuggestionNetwork(nn.Module):
    def __init__(self):
        super(SuggestionNetwork, self).__init__()

        n_suggestions = 128
        dim_actions = 3
        self.suggestion_net = nn.Sequential(
            torch.nn.Linear(dim_actions, 32),
            nn.ReLU(),
            torch.nn.Linear(32, 32),
            nn.ReLU(),
            torch.nn.Linear(32, dim_actions)
        )

    def forward(self, x_vals, noise_smpls):
        #  x_vals = [batch['goal_poses'],  batch['obj_pose'], batch['q0'], batch['collision']]

        goal_poses = x_vals[0]
        obj_pose = x_vals[1]
        q0 = x_vals[2]
        collisions = x_vals[3]
        qg_suggestions = self.suggestion_net(noise_smpls)

        # 32 x n_suggestions x dim_actions
        # I now have to evaluate each one. How?
        # Also, how do I combine information from the goal-classifier and the reachability?
        return qg_suggestions
