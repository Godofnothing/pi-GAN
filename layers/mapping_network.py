import torch
import torch.nn as nn
import torch.nn.functional as F

class MappingNetwork(nn.Module):
    def __init__(
        self, 
        hidden_features, 
        out_features, 
        depth=3, 
        activation=nn.LeakyReLU(negative_slope=0.2)
    ):
        super().__init__()

        self.net = nn.Sequential(
           *sum([
               [nn.Linear(hidden_features, hidden_features), activation]
               for i in range(depth)
            ], [])
        )

        self.to_gamma = nn.Linear(hidden_features, out_features)
        self.to_beta = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = F.normalize(x, dim = -1)
        x = self.net(x)
        return self.to_gamma(x), self.to_beta(x)