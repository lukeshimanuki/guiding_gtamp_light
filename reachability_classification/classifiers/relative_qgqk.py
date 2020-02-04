from torch import nn
import numpy as np
import time
import torch
import torch_scatter


from torch import nn
import numpy as np
import time
import torch
import torch_scatter


class RelativeQgQkGNN(nn.Module):
    def __init__(self, edges, n_key_configs, device, n_msg_passing):
        super(RelativeQgQkGNN, self).__init__()
        self.n_msg_passing = n_msg_passing
        in_channels = 2
        out_channels = 8
        self.x_lin = nn.Sequential(
            torch.nn.Conv2d(1, out_channels, kernel_size=(1, in_channels), stride=1),  # Do this with conv1d
            nn.LeakyReLU())

        self.y_lin = nn.Sequential(
            torch.nn.Conv2d(1, out_channels, kernel_size=(1, in_channels), stride=1),
            nn.LeakyReLU())

        self.th_lin = nn.Sequential(
            torch.nn.Conv2d(1, out_channels, kernel_size=(1, in_channels), stride=1),
            nn.LeakyReLU())

        out_channels = 32
        in_channels = 8 * 3
        self.config_lin = nn.Sequential(
            torch.nn.Conv2d(1, out_channels, kernel_size=(1, in_channels), stride=1),
            nn.LeakyReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1),
            nn.LeakyReLU(),
        )

        in_channels = out_channels
        out_channels = 32
        self.vertex_lin = nn.Sequential(
            torch.nn.Conv2d(1, out_channels, kernel_size=(1, in_channels), stride=1),
            nn.LeakyReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1),
            nn.LeakyReLU(),
        )

        # Edge model
        in_channels = out_channels * 2 + 2 * 2
        out_channels = 32
        self.edge_lin = nn.Sequential(
            torch.nn.Conv2d(1, out_channels, kernel_size=(in_channels, 1), stride=1),
            nn.LeakyReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1),
            nn.LeakyReLU()
        )

        in_channels = 32 + 2
        out_channels_1 = 32
        out_channels = 1

        self.vertex_output_lin = nn.Sequential(
            torch.nn.Conv2d(1, out_channels_1, kernel_size=(in_channels, 1), stride=1),
            nn.LeakyReLU(),
            torch.nn.Conv2d(out_channels_1, out_channels, kernel_size=(1, 1), stride=1),
            nn.LeakyReLU()
            # To visualize the true key config activation I think I will have to replace this with sigmoid. What if I look at the abs of the output here?
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

        ### todo factor the code below
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
        self.device = device
        ### it should repeated in every gnn code

    def concat_at_idx(self, vertex_val_1, vertex_val_2, idx):
        return torch.cat((vertex_val_1[:, :, idx][:, :, None], vertex_val_2[:, :, idx][:, :, None]), -1)

    def reshape_to_feed_to_conv2d(self, n_data, tensor):
        tensor = tensor.reshape((n_data, 1, self.n_nodes, tensor.shape[-1]))
        return tensor

    def compute_msgs(self, v_features, n_data):
        # self.edges[0] - src
        # self.edges[1] - dest
        """
        neighboring_pairs = torch.cat((v_features[:, :, self.edges[0]], v_features[:, :, self.edges[1]]), 1)
        """
        t1 = v_features[:, :, self.edges[0]]
        t2 = v_features[:, :, self.edges[1]]
        t1_dim = t1.shape[1]
        t2_dim = t2.shape[1]
        neighboring_pairs = torch.zeros((n_data, t1_dim + t2_dim, t1.shape[-1])).to(self.device)
        neighboring_pairs[:, 0:t1_dim, :] = t1
        neighboring_pairs[:, t1_dim:, :] = t2

        neighboring_pairs = neighboring_pairs.reshape(
            (n_data, 1, neighboring_pairs.shape[1], neighboring_pairs.shape[-1]))
        msgs = self.edge_lin(neighboring_pairs).squeeze(dim=2)  # 0.4 seconds... but that's cause I am using a cpu
        return msgs

    def get_vertex_features(self, vertices):
        vertex_qk_vals = vertices[:, :, 0:3]
        vertex_qg_vals = vertices[:, :, 3:6]

        # gather the x,y, th vals separately for qkq0 and qkqg
        vertex_qk_qg_xvals = self.concat_at_idx(vertex_qk_vals, vertex_qg_vals, 0)
        vertex_qk_qg_yvals = self.concat_at_idx(vertex_qk_vals, vertex_qg_vals, 1)
        vertex_qk_qg_thvals = self.concat_at_idx(vertex_qk_vals, vertex_qg_vals, 2)

        # compute their features
        n_data = len(vertices)
        vertex_qk_qg_xvals = self.reshape_to_feed_to_conv2d(n_data, vertex_qk_qg_xvals)
        vertex_qk_qg_yvals = self.reshape_to_feed_to_conv2d(n_data, vertex_qk_qg_yvals)
        vertex_qk_qg_thvals = self.reshape_to_feed_to_conv2d(n_data, vertex_qk_qg_thvals)

        vertex_qk_qg_x_features = self.x_lin(vertex_qk_qg_xvals)
        vertex_qk_qg_y_features = self.y_lin(vertex_qk_qg_yvals)
        vertex_qk_qg_th_features = self.th_lin(vertex_qk_qg_thvals)
        qk_qg_config_features = torch.cat(
            (vertex_qk_qg_x_features, vertex_qk_qg_y_features, vertex_qk_qg_th_features), 1)
        qk_qg_config_features = qk_qg_config_features.permute((0, 3, 2, 1))
        vertex_qkqg_feature = self.config_lin(qk_qg_config_features)
        vertex_qkqg_feature = vertex_qkqg_feature.permute((0, 3, 2, 1))
        v_features = self.vertex_lin(vertex_qkqg_feature).squeeze(dim=-1)
        return v_features

    def get_vertex_activations(self, vertices):
        ### First round of vertex feature computation
        # v = np.concatenate([prm_vertices, q0s, qgs, self.collisions[idx]], axis=-1)
        v_features = self.get_vertex_features(vertices)

        collisions = vertices[:, :, 6:]
        collisions = collisions.permute((0, 2, 1))
        v_features = torch.cat((v_features, collisions), 1)
        ##############
        n_data = len(v_features)
        dim_data = v_features.shape[1]
        n_key_configs = v_features.shape[-1]
        v_features = v_features.reshape((n_data, 1, dim_data, n_key_configs))

        final_vertex_output = self.vertex_output_lin(v_features).squeeze()
        n_data = len(vertices)
        if n_data == 1:
            final_vertex_output = final_vertex_output[None, :]
        return final_vertex_output

    def forward(self, vertices):
        # This function computes the node values
        # vertices has shape [n_data, in_channels, n_nodes]
        # edge_index has shape [2, E], top indicating the source and bottom indicating the dest

        final_vertex_output = self.get_vertex_activations(vertices)
        graph_output = self.graph_output_lin(final_vertex_output)
        return graph_output

