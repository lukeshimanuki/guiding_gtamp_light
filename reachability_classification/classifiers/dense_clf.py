import torch
from torch import nn


class DenseReachabilityNet(nn.Module):
    def __init__(self):
        super(DenseReachabilityNet, self).__init__()

        # takes as an input q0, qg, and Ck, and outputs a reachability
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)

        in_channels = 2
        out_channels = 8
        self.x_lin = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU())
        self.y_lin = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU())
        self.th_lin = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU())
        in_channels = 618
        out_channels = 64

        self.collision_lin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(618*2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
        )
        """
        self.collision_lin = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(10, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1)),
            nn.Conv2d(64, 64, kernel_size=(5, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1)),
            nn.Conv2d(64, 32, kernel_size=(3, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1)),
            nn.Flatten()
        )
        """

        self.output_lin = nn.Sequential(
            nn.Linear(64+8*3, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, q0s, qgs, collisions):
        n_data = len(q0s)
        x_concat = torch.cat((q0s[:, 0], qgs[:, 0]), 0).reshape((n_data, 2))
        x_features = self.x_lin(x_concat)
        y_concat = torch.cat((q0s[:, 1], qgs[:, 1]), 0).reshape((n_data, 2))
        y_features = self.y_lin(y_concat)
        th_concat = torch.cat((q0s[:, -1], qgs[:, -1]), 0).reshape((n_data, 2))
        th_features = self.th_lin(th_concat)

        q0qg_features = torch.cat((x_features, y_features, th_features), -1).reshape(n_data, 8 * 3)
        collision_features = self.collision_lin(collisions)
        everything = torch.cat((q0qg_features, collision_features), -1)
        output = self.output_lin(everything)

        return output


