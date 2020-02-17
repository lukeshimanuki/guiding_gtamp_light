import torch
from torch import nn


class DenseIMLETorch(nn.Module):
    def __init__(self):
        super(DenseIMLETorch, self).__init__()

        in_channels = 20
        out_channels = 512
        final_out_channels = 1
        self.x_lin = nn.Sequential(
            torch.nn.Conv2d(1, out_channels, kernel_size=(1, in_channels), stride=1),
            nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1),
            nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1),
            nn.ReLU(),
            torch.nn.Conv2d(out_channels, final_out_channels, kernel_size=(1, 1), stride=1),
            nn.ReLU()
        )
        """
        torch.nn.Conv2d(1, out_channels, kernel_size=(1, 1), stride=1),  # Do this with conv1d
        nn.ReLU(),
        torch.nn.Conv2d(1, out_channels, kernel_size=(1, 1), stride=1),  # Do this with conv1d
        nn.ReLU(),
        torch.nn.Conv2d(1, final_out_channels, kernel_size=(1, 1), stride=1),  # Do this with conv1d
        nn.ReLU()
        """

        n_konfs = 618
        self.output_lin = nn.Sequential(
            torch.nn.Linear(n_konfs*final_out_channels+n_konfs, 32),
            nn.ReLU(),
            torch.nn.Linear(32, 3)
        )

    def forward(self, x_vals, noise_smpls):
        x_features = self.x_lin(x_vals)
        x_features = x_features.flatten(start_dim=1)

        x_noise = torch.cat((x_features, noise_smpls), -1)
        output = self.output_lin(x_noise)
        return output

