import torch
from torch import nn
from torch_geometric.utils import scatter_
import torch.nn.functional as F
import time
import torch
import torch_scatter


class GNNReachabilityNet(nn.Module):
    def __init__(self, edges, n_key_configs):
        super(GNNReachabilityNet, self).__init__()

        # Vertex model. Currently takes all xyth, col into account
        in_channels = 11
        out_channels = 32
        self.x_lin = nn.Sequential(
            torch.nn.Conv2d(1, out_channels, kernel_size=(1, in_channels), stride=1),
            nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1),
            nn.ReLU()
        )

        # Edge model
        in_channels = out_channels * 2
        out_channels = 32
        self.edge_lin = nn.Sequential(
            torch.nn.Conv2d(1, out_channels, kernel_size=(in_channels, 1), stride=1),
            nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1),
            nn.ReLU()
        )

        in_channels = 32
        out_channels = 1
        self.vertex_output_lin = nn.Sequential(
            torch.nn.Conv2d(1, out_channels, kernel_size=(in_channels, 1), stride=1),
            nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1),
            nn.ReLU()
        )

        n_nodes = n_key_configs
        self.graph_output_lin = nn.Sequential(
            torch.nn.Linear(n_nodes, 1),
            nn.Sigmoid()
        )

        self.edges = torch.from_numpy(edges)
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
        return self.graph_output_lin(final_vertex_output)

"""
x = torch.rand(618, 11)  # but then how do I create multiple data points?
edge_idx = torch.tensor([[0, 1, 2],  # it's always bidrectional
                         [1, 2, 1]],
                        dtype=torch.long)  # 2 x n_edges, top indicating the src and bottom indicating the dest

data = {'vtxs': x}
data_list = [data] * 10000
loader = torch.utils.data.DataLoader(data_list, batch_size=32, shuffle=True, num_workers=2)
net = GNNReachabilityNet()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
device = torch.device('cpu')


def train(epoch):
    net.to(device)
    net.train()
    total_loss = 0
    for data in loader:
        data['vtxs'].to(device)
        optimizer.zero_grad()
        out = net(data['vtxs'])
        loss = ((out - data.y).pow(2)).mean()
        loss.backward()
        optimizer.step()

    import pdb;
    pdb.set_trace()
    return total_loss / len(train_loader.dataset)


train(1)

# I need to put it through a dense net to produce results

# How do I do:
#   1. Aggregating all the outputs for final prediction? Take the output from net, and put it through a dense net
#   2. Multiple rounds of msg passing? I guess I take the output and put it through the net again? This requires
#       maintaining the same output value. What I perhaps want to do is to create another network that now takes the
#       graph data.
"""
