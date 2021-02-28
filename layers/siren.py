
import torch
from torch import nn
import torch.nn.functional as F

import math

from .activations import Sin

class Siren(nn.Module):
    def __init__(
            self, 
            dim_in, 
            dim_out, 
            w0 = 1., 
            c = 6., 
            is_first = False, 
            use_bias = True, 
            activation = None,
            initializer = "uniform"
        ):

        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.initializer = initializer

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sin(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)

        if self.initializer == "uniform":
            weight.uniform_(-w_std, w_std)
            if bias is not None:
                bias.uniform_(-w_std, w_std)
        elif self.initializer == "normal":
            weight.normal_(-w_std, w_std)
            if bias is not None:
                bias.normal_(-w_std, w_std)

    def forward(self, x, gamma = None, beta = None):
        out =  F.linear(x, self.weight, self.bias)

        # FiLM modulation
        if gamma is not None:
            out = out * gamma

        if beta is not None:
            out = out + beta

        out = self.activation(out)
        return out