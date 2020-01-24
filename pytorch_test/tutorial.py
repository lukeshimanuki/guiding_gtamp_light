import torch
from torch import nn



# let's first try to define the f function
# In PyTorch, images are represented as [channels, height, width], so a color image would be [3, 256, 256].
m = nn.Conv1d(4, 32, 1, stride=1)   # in_channels, out_channels, kernel_size
m2 = nn.Conv1d(32, 32, 1, stride=1)
input = torch.randn(1, 4, 1) # (n_data, n_channel, width)

nn.ReLU()

## Try to define f_theta for q0 and qg


x_detector = nn.Conv1d(2, 32, 1, stride=1)  # n_channels, n_output, width
y_detector = nn.Conv1d(2, 32, 1, stride=1)  # n_channels, n_output, width
th_detector = nn.Conv1d(4, 32, 1, stride=1)  # n_channels, n_output, width

vertex_input = torch.randn(1, 10, 100)  # case when we have 100 vertices: [q0 qk ck] - 4 4 2 -> 14 dimensions

vertex_input[0, :, 0] = vertex_input[0, :, 1]
x_val = x_detector(vertex_input[:, 0:2, :])  # SO fucking intuitive. I am loving pytorch
y_val = y_detector(vertex_input[:, 2:4, :])
th_val = th_detector(vertex_input[:, 4:8, :])
f_detector = nn.Conv1d(32*3, 64, 1, stride=1)

f_detector(torch.cat((x_val, y_val, th_val), 1))




import pdb;pdb.set_trace()
