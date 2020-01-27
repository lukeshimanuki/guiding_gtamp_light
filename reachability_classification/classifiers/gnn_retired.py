import torch
from torch import nn
from torch_geometric.nn import MessagePassing, global_max_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import TUDataset
import torch.nn.functional as F
import time


class FirstGraphNet(MessagePassing):
    def __init__(self):
        super(FirstGraphNet, self).__init__(aggr='mean', flow='source_to_target')
        """
        in_channels = 2
        out_channels = 32
        self.x_lin = torch.nn.Linear(in_channels, out_channels)
        self.y_lin = torch.nn.Linear(in_channels, out_channels)

        in_channels = 2
        self.th_lin = torch.nn.Linear(in_channels, out_channels)
        """

        in_channels = 11
        out_channels = 64
        self.vertex_lin = torch.nn.Linear(in_channels, out_channels)

        in_channels = out_channels * 2 + 4
        out_channels = 128
        self.edge_lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, vertex_input, edge_index):
        # This function computes the node values
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        """
        q_k_q0_x_val = self.x_lin(vertex_input[:, [0, 4]])
        q0_y_val = self.x_lin(vertex_input[:, [1, 5]])
        y_val = self.y_lin(vertex_input[:, 1,5])
        th_val = self.th_lin(vertex_input[:, 4:8])
        xyth = torch.cat((x_val, y_val, th_val), -1)
        """
        import pdb;pdb.set_trace()
        f_detector = self.vertex_lin(vertex_input)
        return self.propagate(edge_index, x=f_detector, col=vertex_input[:, 9:])

    def message(self, x_i, x_j, col_i, col_j):
        # This function computes the msg from x_j to its neighboring node based on the output from the forward function
        # Convention - *_i: destination *_j: source
        # Each x_i is n_edges x n_features, using the convention above
        edge_input = torch.cat((x_i, x_j, col_i, col_j), -1)
        msg = self.edge_lin(edge_input)
        return msg

    def update(self, aggr_out):
        # This function updates node embeddings for every node
        # aggr_out has shape [N, out_channels]
        return aggr_out


class FinalGraphNet(MessagePassing):
    def __init__(self):
        super(FinalGraphNet, self).__init__(aggr='mean', flow='source_to_target')
        in_channels = 128
        out_channels = 64
        self.x_lin = torch.nn.Linear(in_channels, out_channels)

        in_channels = 64 * 2
        self.edge_lin = torch.nn.Linear(in_channels, out_channels)

        in_channels = 64
        out_channels = 1
        self.output_lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, vertex_input, edge_index):
        # This function computes the node values
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        x_val = self.x_lin(vertex_input)
        return self.propagate(edge_index, size=(x_val.size(0), x_val.size(0)), x=x_val)

    def message(self, x_i, x_j):
        # This function computes the msg from x_j to its neighboring node based on the output from the forward function
        # Convention - *_i: destination *_j: source
        # Can take any argument which was initially passed to propagate()
        # I am supposed to construct messages at each edge
        edge_input = torch.cat((x_i, x_j), -1)
        msg = self.edge_lin(edge_input)
        return msg

    def update(self, aggr_out):
        # This function updates node embeddings for every node
        # aggr_out has shape [N, out_channels]
        output = self.output_lin(aggr_out)
        return output


class GNNReachabilityNet(torch.nn.Module):
    def __init__(self):
        super(GNNReachabilityNet, self).__init__()
        self.firstnet = FirstGraphNet()
        self.finalnet = FinalGraphNet()

    def forward(self, data, batch):
        x, edge_index = data.x, data.edge_idx
        # What do I do if I have multiple inputs?
        import pdb;pdb.set_trace()
        first_output = self.firstnet(x, edge_index)
        final_output = self.finalnet(first_output, edge_index)
        # The way to read batch:
        #   [0,0,0,0,0, 1, 1,1,1,1] means that your first graph has 5 nodes,
        #   and these are the output from the five nodes.
        #   What I want to do is to take the local pooling, make the graph smaller, and then pass it through a dense network.
        return global_max_pool(final_output, batch)


x = torch.rand(3, 11)  # but then how do I create multiple data points?
edge_idx = torch.tensor([[0, 1, 1, 2],
                         [1, 0, 2, 1]],
                        dtype=torch.long)  # 2 x n_edges, top indicating the src and bottom indicating the dest


data = Data(edge_idx=edge_idx, x=x, y=0.0)
data_list = [data] * 10000
loader = DataLoader(data_list, batch_size=2, shuffle=True)
net = GNNReachabilityNet()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
device = torch.device('cpu')


def train(epoch):
    net.to(device)
    net.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = net(data, data.batch)
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
