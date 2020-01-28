import torch
from torch import nn
from torch_geometric.utils import scatter_
import torch.nn.functional as F
import time
import torch
import torch_scatter


class GNNReachabilityNet(nn.Module):
    def __init__(self, edges, n_key_configs, device):
        super(GNNReachabilityNet, self).__init__()

        # Vertex model. Currently takes all xyth, col into account
        in_channels = 11
        out_channels = 32
        self.x_lin = nn.Sequential(
            torch.nn.Conv2d(1, out_channels, kernel_size=(1, in_channels), stride=1),
            nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1),
            nn.LeakyReLU()
        )

        # Edge model
        in_channels = out_channels * 2
        out_channels = 32
        self.edge_lin = nn.Sequential(
            torch.nn.Conv2d(1, out_channels, kernel_size=(in_channels, 1), stride=1),
            nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1),
            nn.LeakyReLU()
        )

        in_channels = 32
        out_channels_1 = 32
        out_channels = 1

        self.vertex_output_lin = nn.Sequential(
            torch.nn.Conv2d(1, out_channels_1, kernel_size=(in_channels, 1), stride=1),
            nn.ReLU(),
            torch.nn.Conv2d(out_channels_1, out_channels, kernel_size=(1, 1), stride=1),
            nn.LeakyReLU()
        )

        n_nodes = n_key_configs
        self.graph_output_lin = nn.Sequential(
            torch.nn.Linear(n_nodes, 1),
            nn.Sigmoid()
        )

        self.edges = torch.from_numpy(edges).to(device)
        self.n_nodes = n_key_configs

    def forward(self, vertices):
        # This function computes the node values
        # vertices has shape [n_data, in_channels, n_nodes]
        # edge_index has shape [2, E], top indicating the source and bottom indicating the dest

        vertices = vertices.reshape((vertices.shape[0], 1, self.n_nodes, 11))
        v_features = self.x_lin(vertices).squeeze()

        neighboring_pairs = torch.cat((v_features[:, :, self.edges[0]], v_features[:, :, self.edges[1]]), 1)
        neighboring_pairs = neighboring_pairs.reshape((vertices.shape[0], 1, neighboring_pairs.shape[1],
                                                       neighboring_pairs.shape[-1]))
        msgs = self.edge_lin(neighboring_pairs).squeeze()

        agg = torch_scatter.scatter_mean(msgs, self.edges[1], dim=-1)

        agg = agg.reshape((agg.shape[0], 1, agg.shape[1], agg.shape[2]))
        final_vertex_output = self.vertex_output_lin(agg).squeeze()
        graph_output = self.graph_output_lin(final_vertex_output)
        return graph_output
