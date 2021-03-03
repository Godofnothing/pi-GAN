import torch
import torch.nn as nn
import torch.nn.functional as F

from .siren import Siren
from .siren_net import SirenMLP
from .mapping_network import MappingNetwork
from utils import get_image_from_nerf_model

class SirenGenerator(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_hidden,
        mapping_network_kw,
        siren_mlp_kw,
        activation=nn.LeakyReLU(negative_slope=0.2)
    ):
        super().__init__()

        self.mapping = MappingNetwork(
            dim_hidden=dim_input,
            dim_out=dim_hidden,
            activation=activation,
            **mapping_network_kw
        )

        self.siren = SirenMLP(
            dim_in = 3,
            dim_hidden = dim_hidden,
            dim_out = dim_hidden,
            **siren_mlp_kw
        )

        self.to_alpha = nn.Linear(
            dim_hidden, 1
        )

        self.to_rgb_siren = Siren(
            dim_in=dim_hidden,
            dim_out=dim_hidden
        )

        self.to_rgb = nn.Linear(
            dim_hidden, 3
        )

    def forward(self, latent, coors, batch_size=8192):
        gamma, beta = self.mapping(latent)

        outs = []
        for coor in coors.split(batch_size):
            gamma_, beta_ = map(lambda t: t[None, ...], (gamma, beta))
            x = self.siren(coor, gamma_, beta_)
            alpha = self.to_alpha(x)

            x = self.to_rgb_siren(x, gamma, beta)
            rgb = self.to_rgb(x)
            out = torch.cat((rgb, alpha), dim = -1)
            outs.append(out)

        return torch.cat(outs)

class Generator(nn.Module):
    def __init__(
        self,
        image_size,
        dim_input,
        dim_hidden,
        mapping_network_kw,
        siren_mlp_kw,
        activation=nn.LeakyReLU(negative_slope=0.2)
    ):
        super().__init__()
        self.dim_input = dim_input
        self.image_size = image_size

        self.nerf_model = SirenGenerator(
            dim_input = dim_input,
            dim_hidden = dim_hidden,
            activation=activation,
            mapping_network_kw=mapping_network_kw,
            siren_mlp_kw=siren_mlp_kw
        )

    def set_image_size(self, image_size):
        self.image_size = image_size

    def forward(self, latents):
        image_size = self.image_size

        generated_images = get_image_from_nerf_model(
            self.nerf_model,
            latents,
            image_size,
            image_size
        )

        return generated_images