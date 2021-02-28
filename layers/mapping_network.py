import torch
import torch.nn as nn
import torch.nn.functional as F

class MappingNetwork(nn.Module):
    def __init__(self, dim_hidden, dim_out, depth=3, activation=nn.LeakyReLU()):
        super().__init__()

        self.net = nn.Sequential(
           *sum([
               [nn.Linear(dim_hidden, dim_hidden), activation]
               for i in range(depth)
            ], [])
        )

        self.to_gamma = nn.Linear(dim_hidden, dim_out)
        self.to_beta = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = F.normalize(x, dim = -1)
        x = self.net(x)
        return self.to_gamma(x), self.to_beta(x)