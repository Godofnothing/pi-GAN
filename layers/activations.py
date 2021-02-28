import torch
from torch import nn as nn

class Sin(nn.Module):
    
    def __init__(self, w = 1.):
        super().__init__()
        self.w = w

    def forward(self, x):
        return torch.sin(self.w * x)