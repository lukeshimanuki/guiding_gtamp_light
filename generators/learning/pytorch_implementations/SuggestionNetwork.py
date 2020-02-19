import torch
from torch import nn


class SuggestionNetwork(nn.Module):
    def __init__(self):
        super(SuggestionNetwork, self).__init__()

        n_suggestions = 128
        dim_actions = 4
        self.suggestion_net = nn.Sequential(
            torch.nn.Linear(dim_actions, 32),
            nn.ReLU(),
            torch.nn.Linear(32, 32),
            nn.ReLU(),
            torch.nn.Linear(dim_actions*n_suggestions, 32)
        )

    def forward(self, x_vals, noise_smpls):
        x_features = self.x_lin(x_vals)
        x_features = x_features.flatten(start_dim=1)

        x_noise = torch.cat((x_features, noise_smpls), -1)
        output = self.output_lin(x_noise)
        return output

