import torch
from torch import nn


class BaseDiscriminator(nn.Module):
    def __init__(self, dim_konf, atype):
        nn.Module.__init__(self)
        self.dim_konf = dim_konf
        self.atype = atype

    def forward(self, action, konf, pose):
        raise NotImplementedError


class BaseGenerator(nn.Module):
    def __init__(self, dim_konf, atype):
        nn.Module.__init__(self)
        self.dim_konf = dim_konf
        self.atype = atype

    def forward(self, konf, pose, noise):
        raise NotImplementedError
