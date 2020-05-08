import torch
import torch_scatter
from models import BaseGenerator, BaseDiscriminator
import numpy as np
import pickle


def get_edges():
    prm_vertices, prm_edges = pickle.load(open('prm.pkl', 'r'))

    edges = [[], []]
    for src_idx, _ in enumerate(prm_vertices):
        neighbors = list(prm_edges[src_idx])
        n_edges = len(neighbors)
        edges[0] += [src_idx] * n_edges
        edges[1] += neighbors
    return torch.from_numpy(np.array(edges))


class GNNDiscriminator(BaseDiscriminator):
    def __init__(self, dim_konf, dim_actions, device):
        BaseDiscriminator.__init__(self, dim_konf)
        self.edges = get_edges().to(device)

        n_hidden = 32
        self.first_features = \
            torch.nn.Sequential(
                torch.nn.Conv2d(1, n_hidden, kernel_size=(1, self.dim_konf + 8)),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(n_hidden, n_hidden, kernel_size=(1, 1)),
                torch.nn.LeakyReLU()
            )

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(n_hidden, n_hidden, kernel_size=(1, 1)),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(n_hidden, n_hidden, kernel_size=(1, 1)),
            torch.nn.LeakyReLU()
        )

        self.value = \
            torch.nn.Sequential(
                torch.nn.Linear(32 * 154, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1)
            )

    def forward(self, action, konf, pose):
        action_expanded = action.unsqueeze(1).repeat((1, 618, 1)).unsqueeze(-1)
        robot_curr_pose = pose[:, -4:]
        robot_curr_pose_expanded = robot_curr_pose.unsqueeze(1).repeat((1, 618, 1)).unsqueeze(-1)
        concat = torch.cat([action_expanded, robot_curr_pose_expanded, konf], dim=2)
        concat = concat.reshape((concat.shape[0], concat.shape[-1], concat.shape[1], concat.shape[2]))

        features = self.first_features(concat)
        features = features.squeeze()
        paired_feature_values = features[:, :, self.edges[1]]
        indices_of_neighboring_nodes = self.edges[0]

        vertex_values = torch_scatter.scatter_mean(paired_feature_values, indices_of_neighboring_nodes, dim=-1)

        n_passes = 2
        for _ in range(n_passes):
            vertex_values = vertex_values.view(
                (vertex_values.shape[0], vertex_values.shape[1], vertex_values.shape[2], 1))
            vertex_values = self.features(vertex_values).squeeze()
            paired_feature_values = vertex_values[:, :, self.edges[1]]
            vertex_values = torch_scatter.scatter_mean(paired_feature_values, indices_of_neighboring_nodes, dim=-1)
        features = features.view((features.shape[0], features.shape[1], features.shape[2], 1))
        features = torch.nn.MaxPool2d(kernel_size=(2, 1))(features)
        features = torch.nn.MaxPool2d(kernel_size=(2, 1))(features)
        features = features.view((features.shape[0], features.shape[1]*features.shape[2]))

        value = self.value(features)
        return value


class GNNGenerator(BaseGenerator):
    def __init__(self, dim_konf, dim_data, device):
        self.edges = get_edges().to(device)
        BaseGenerator.__init__(self, dim_konf)
        n_hidden = 32
        self.first_features = \
            torch.nn.Sequential(
                torch.nn.Conv2d(1, n_hidden, kernel_size=(1, self.dim_konf + 4)),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(n_hidden, n_hidden, kernel_size=(1, 1)),
                torch.nn.LeakyReLU()
            )

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(n_hidden, n_hidden, kernel_size=(1, 1)),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(n_hidden, n_hidden, kernel_size=(1, 1)),
            torch.nn.LeakyReLU()
        )

        self.value = \
            torch.nn.Sequential(
                torch.nn.Linear(32 * 154, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, dim_data))

    def forward(self, konf, pose, noise):
        robot_curr_pose = pose[:, -4:]
        robot_curr_pose_expanded = robot_curr_pose.unsqueeze(1).repeat((1, 618, 1)).unsqueeze(-1)
        concat = torch.cat([robot_curr_pose_expanded, konf], dim=2)
        concat = concat.reshape((concat.shape[0], concat.shape[-1], concat.shape[1], concat.shape[2]))

        features = self.first_features(concat)
        features = features.squeeze()
        paired_feature_values = features[:, :, self.edges[1]]
        indices_of_neighboring_nodes = self.edges[0]

        vertex_values = torch_scatter.scatter_mean(paired_feature_values, indices_of_neighboring_nodes, dim=-1)

        n_passes = 2
        for _ in range(n_passes):
            vertex_values = vertex_values.view(
                (vertex_values.shape[0], vertex_values.shape[1], vertex_values.shape[2], 1))
            vertex_values = self.features(vertex_values).squeeze()
            paired_feature_values = vertex_values[:, :, self.edges[1]]
            vertex_values = torch_scatter.scatter_mean(paired_feature_values, indices_of_neighboring_nodes, dim=-1)

        features = features.view((features.shape[0], features.shape[1], features.shape[2], 1))
        features = torch.nn.MaxPool2d(kernel_size=(2, 1))(features)
        features = torch.nn.MaxPool2d(kernel_size=(2, 1))(features)
        features = features.view((features.shape[0], features.shape[1] * features.shape[2]))
        value = self.value(features)

        return value
