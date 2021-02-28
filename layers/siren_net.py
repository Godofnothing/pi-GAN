import torch
import torch.nn as nn
import torch.nn.functional as F

from .siren import Siren

class SirenMLP(nn.Module):
    def __init__(self, 
        dim_in, 
        dim_hidden, 
        dim_out, 
        num_layers, 
        w0 = 1., 
        w0_initial = 30., 
        use_bias = True, 
        final_activation = None,
        initializer = "uniform"
    ):
        super(SirenMLP, self).__init__()

        assert num_layers >= 1

        self.layers = nn.ModuleList(
            [
               Siren(
                   dim_in=dim_in if idx == 0 else dim_hidden,
                   dim_out=dim_hidden,
                   w0=w0_initial if idx == 0 else w0,
                   use_bias=use_bias,
                   is_first=(idx == 0)
               )
               for idx in range(num_layers) 
            ]
        )

        self.last_layer = Siren(
            dim_in=dim_hidden, 
            dim_out=dim_out, 
            w0=w0, 
            use_bias=use_bias, 
            activation=final_activation,
            initializer=initializer
        )

    def forward(self, x, gamma, beta):
        for layer in self.layers:
            x = layer(x, gamma, beta)
        return self.last_layer(x)