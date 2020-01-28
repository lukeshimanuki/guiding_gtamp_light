import torch
from torch import nn
from torch_geometric.utils import scatter_
import torch.nn.functional as F
import time
import numpy as np
import torch
import torch_scatter


class SimpleMultiplePassGNNReachabilityNet(nn.Module):
    def __init__(self, edges, n_key_configs, device):
        super(SimpleMultiplePassGNNReachabilityNet, self).__init__()

        # Vertex model. Currently takes all xyth, col into account
        in_channels = 11
        out_channels = 32
        self.x_lin = nn.Sequential(
            torch.nn.Conv2d(1, out_channels, kernel_size=(1, in_channels), stride=1),
            nn.LeakyReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1),
            nn.LeakyReLU()
        )

        # Edge model
        in_channels = out_channels * 2
        out_channels = 32
        self.edge_lin = nn.Sequential(
            torch.nn.Conv2d(1, out_channels, kernel_size=(in_channels, 1), stride=1),
            nn.LeakyReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1),
            nn.LeakyReLU()
        )

        in_channels = 32
        out_channels_1 = 32
        out_channels = 1

        self.vertex_output_lin = nn.Sequential(
            torch.nn.Conv2d(1, out_channels_1, kernel_size=(in_channels, 1), stride=1),
            nn.LeakyReLU(),
            torch.nn.Conv2d(out_channels_1, out_channels, kernel_size=(1, 1), stride=1),
            nn.LeakyReLU()
        )

        n_nodes = n_key_configs
        self.graph_output_lin = nn.Sequential(
            torch.nn.Linear(n_nodes, 1),
            nn.Sigmoid()
        )

        in_channels = 32
        out_channels = 32
        self.x_lin_after_first_round = nn.Sequential(
            torch.nn.Conv2d(1, out_channels, kernel_size=(in_channels, 1), stride=1),
            nn.LeakyReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1),
            nn.LeakyReLU()
        )

        non_duplicate_edges = [[], []]
        duplicate_checker = []
        for src, dest in zip(edges[0], edges[1]):
            to_add = set((src, dest))
            if to_add not in duplicate_checker:
                duplicate_checker.append(to_add)
                non_duplicate_edges[0].append(src)
                non_duplicate_edges[1].append(dest)

        self.edges = torch.from_numpy(np.array(non_duplicate_edges)).to(device)
        self.dest_edges = torch.from_numpy(np.hstack((non_duplicate_edges[1], non_duplicate_edges[0]))).to(device)
        self.n_nodes = n_key_configs

    def compute_msgs(self, v_features, n_data):
        # self.edges[0] - src
        # self.edges[1] - dest
        neighboring_pairs = torch.cat((v_features[:, :, self.edges[0]], v_features[:, :, self.edges[1]]), 1)
        neighboring_pairs = neighboring_pairs.reshape((n_data, 1, neighboring_pairs.shape[1],
                                                       neighboring_pairs.shape[-1]))
        msgs = self.edge_lin(neighboring_pairs).squeeze()  # 0.4 seconds... but that's cause I am using a cpu
        return msgs

    def forward(self, vertices):
        # This function computes the node values
        # vertices has shape [n_data, in_channels, n_nodes]
        # edge_index has shape [2, E], top indicating the source and bottom indicating the dest
        vertices = vertices.reshape((vertices.shape[0], 1, self.n_nodes, 11))
        v_features = self.x_lin(vertices).squeeze()

        msgs = self.compute_msgs(v_features, len(vertices))
        msgs = msgs.repeat((1, 1, 2))
        new_vertex = torch_scatter.scatter_mean(msgs, self.dest_edges, dim=-1)
        new_vertex = new_vertex[:, None, :, :]

        ##### msg passing
        n_msg_passing = 5
        for i in range(n_msg_passing):
            vertices_after_first_round = self.x_lin_after_first_round(new_vertex).squeeze()
            msgs = self.compute_msgs(vertices_after_first_round, len(vertices))
            msgs = msgs.repeat((1, 1, 2))
            residual = torch_scatter.scatter_mean(msgs, self.dest_edges, dim=-1)
            residual = residual[:, None, :, :]
            new_vertex = new_vertex + residual
        ##### end of msg passing

        # Final round of output
        # new_vertex = new_vertex, 1, new_vertex.shape[1], new_vertex.shape[2]))
        final_vertex_output = self.vertex_output_lin(new_vertex).squeeze()
        graph_output = self.graph_output_lin(final_vertex_output)
        return graph_output
