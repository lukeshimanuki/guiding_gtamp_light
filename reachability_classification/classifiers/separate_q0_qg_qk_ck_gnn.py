import torch
from torch import nn
from torch_geometric.utils import scatter_
import torch.nn.functional as F
import time
import torch
import torch_scatter


class Separateq0qgqkckGNNReachabilityNet(nn.Module):
    def __init__(self, edges, n_key_configs, device):
        super(Separateq0qgqkckGNNReachabilityNet, self).__init__()

        # Vertex model. Currently takes all xyth, col into account
        in_channels = 11
        out_channels = 32
        """
        self.x_lin = nn.Sequential(
            torch.nn.Conv2d(1, out_channels, kernel_size=(1, in_channels), stride=1),
            nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1),
            nn.LeakyReLU()
        )
        """
        in_channels = 2
        out_channels = 8
        self.x_lin = nn.Sequential(
            torch.nn.Conv2d(1, out_channels, kernel_size=(1, in_channels), stride=1), # Do this with conv1d
            nn.LeakyReLU())

        self.y_lin = nn.Sequential(
            torch.nn.Conv2d(1, out_channels, kernel_size=(1, in_channels), stride=1),
            nn.LeakyReLU())

        self.th_lin = nn.Sequential(
            torch.nn.Conv2d(1, out_channels, kernel_size=(1, in_channels), stride=1),
            nn.LeakyReLU())

        out_channels = 32
        self.config_lin = nn.Sequential(
            torch.nn.Conv2d(8*3, out_channels, kernel_size=(1, 1), stride=1),
            nn.LeakyReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1),
            nn.LeakyReLU(),
        )

        in_channels = out_channels*2
        out_channels = 32
        self.vertex_lin = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=1),
            nn.LeakyReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1),
            nn.LeakyReLU(),
        )

        # Edge model
        in_channels = out_channels * 2 + 2*2
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

    def concat_at_idx(self, vertex_val_1, vertex_val_2, idx):
        return torch.cat((vertex_val_1[:, :, idx][:, :, None], vertex_val_2[:, :, idx][:, :, None]), -1)

    def reshape_to_feed_to_conv2d(self, n_data, tensor):
        tensor = tensor.reshape((n_data, 1, self.n_nodes, tensor.shape[-1]))
        return tensor

    def forward(self, vertices):
        # This function computes the node values
        # vertices has shape [n_data, in_channels, n_nodes]
        # edge_index has shape [2, E], top indicating the source and bottom indicating the dest

        # v = np.concatenate([prm_vertices, q0s, qgs, self.collisions[idx]], axis=-1)
        vertex_qk_vals = vertices[:, :, 0:3]
        vertex_q0_vals = vertices[:, :, 3:6]
        vertex_qg_vals = vertices[:, :, 6:9]

        # gather the x,y, th vals separately for qkq0 and qkqg
        vertex_qk_q0_xvals = self.concat_at_idx(vertex_qk_vals, vertex_q0_vals, 0)
        vertex_qk_q0_yvals = self.concat_at_idx(vertex_qk_vals, vertex_q0_vals, 1)
        vertex_qk_q0_thvals = self.concat_at_idx(vertex_qk_vals, vertex_q0_vals, 2)
        vertex_qk_qg_xvals = self.concat_at_idx(vertex_qk_vals, vertex_qg_vals, 0)
        vertex_qk_qg_yvals = self.concat_at_idx(vertex_qk_vals, vertex_qg_vals, 1)
        vertex_qk_qg_thvals = self.concat_at_idx(vertex_qk_vals, vertex_qg_vals, 2)

        # compute their features
        n_data = len(vertices)
        vertex_qk_q0_xvals = self.reshape_to_feed_to_conv2d(n_data, vertex_qk_q0_xvals)
        vertex_qk_q0_yvals = self.reshape_to_feed_to_conv2d(n_data, vertex_qk_q0_yvals)
        vertex_qk_q0_thvals = self.reshape_to_feed_to_conv2d(n_data, vertex_qk_q0_thvals)
        vertex_qk_qg_xvals = self.reshape_to_feed_to_conv2d(n_data, vertex_qk_qg_xvals)
        vertex_qk_qg_yvals = self.reshape_to_feed_to_conv2d(n_data, vertex_qk_qg_yvals)
        vertex_qk_qg_thvals = self.reshape_to_feed_to_conv2d(n_data, vertex_qk_qg_thvals)

        vertex_qk_q0_x_features = self.x_lin(vertex_qk_q0_xvals)
        vertex_qk_q0_y_features = self.y_lin(vertex_qk_q0_yvals)
        vertex_qk_q0_th_features = self.th_lin(vertex_qk_q0_thvals)
        qk_q0_config_features = torch.cat(
            (vertex_qk_q0_x_features, vertex_qk_q0_y_features, vertex_qk_q0_th_features), 1)

        vertex_qk_qg_x_features = self.x_lin(vertex_qk_qg_xvals)
        vertex_qk_qg_y_features = self.y_lin(vertex_qk_qg_yvals)
        vertex_qk_qg_th_features = self.th_lin(vertex_qk_qg_thvals)
        qk_qg_config_features = torch.cat(
            (vertex_qk_qg_x_features, vertex_qk_qg_y_features, vertex_qk_qg_th_features), 1)

        vertex_qkq0_feature = self.config_lin(qk_q0_config_features)
        vertex_qkqg_feature = self.config_lin(qk_qg_config_features)

        config_features = torch.cat((vertex_qkq0_feature, vertex_qkqg_feature), 1)
        v_features = self.vertex_lin(config_features).squeeze()

        collisions = vertices[:, :, 9:]
        collisions = collisions.permute((0, 2, 1))
        v_features = torch.cat((v_features, collisions), 1)

        # below cat line takes 0.5 seconds. I can reduce it by half by using the fact that graph is always bidrectional
        neighboring_pairs = torch.cat((v_features[:, :, self.edges[0]], v_features[:, :, self.edges[1]]), 1)
        neighboring_pairs = neighboring_pairs.reshape((vertices.shape[0], 1, neighboring_pairs.shape[1], neighboring_pairs.shape[-1]))
        msgs = self.edge_lin(neighboring_pairs).squeeze()

        agg = torch_scatter.scatter_mean(msgs, self.edges[1], dim=-1)

        agg = agg.reshape((agg.shape[0], 1, agg.shape[1], agg.shape[2]))
        final_vertex_output = self.vertex_output_lin(agg).squeeze()
        graph_output = self.graph_output_lin(final_vertex_output)
        return graph_output
