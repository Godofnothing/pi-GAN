import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .coord_conv import  CoordConv

class DiscriminatorBlock(nn.Module):
    def __init__(
        self, 
        dim, 
        dim_out, 
        activation=nn.LeakyReLU()
    ):
        super().__init__()
        self.res = CoordConv(dim, dim_out, kernel_size = 1, stride = 2)

        self.net = nn.Sequential(
            CoordConv(dim, dim_out, kernel_size = 3, padding = 1),
            activation,
            CoordConv(dim_out, dim_out, kernel_size = 3, padding = 1),
            activation
        )

        self.down = nn.AvgPool2d(2)

    def forward(self, x):
        res = self.res(x)
        x = self.net(x)
        x = self.down(x)
        x = x + res
        # print(x.shape)
        return x

class Discriminator(nn.Module):
    def __init__(
        self,
        image_size,
        init_chan = 64,
        max_chan = 400,
        init_resolution = 32,
        add_layer_iters = 10000,
        discriminator_block_activation = nn.LeakyReLU(),
        from_rgb_activation = nn.LeakyReLU(),
    ):
        super().__init__()
        resolutions = math.log2(image_size)
        self.init_resolution = init_resolution
        assert resolutions.is_integer(), 'image size must be a power of 2'
        assert math.log2(init_resolution).is_integer(), 'initial resolution must be power of 2'

        resolutions = int(resolutions)
        layers = resolutions - 1

        chans = list(reversed(list(map(lambda t: 2 ** (11 - t), range(layers)))))
        chans = list(map(lambda n: min(max_chan, n), chans))
        chans = [init_chan, *chans]
        final_chan = chans[-1]

        self.from_rgb_layers = nn.ModuleList([])
        self.layers = nn.ModuleList([])
        self.image_size = image_size
        self.resolutions = list(map(lambda t: 2 ** (resolutions - t), range(layers)))

        for resolution, in_chan, out_chan in zip(self.resolutions, chans[:-1], chans[1:]):

            from_rgb_layer = nn.Sequential(
                CoordConv(3, in_chan, kernel_size = 1),
                from_rgb_activation
            ) if resolution >= init_resolution else None

            self.from_rgb_layers.append(from_rgb_layer)

            self.layers.append(
                DiscriminatorBlock(
                    dim=in_chan,
                    dim_out=out_chan,
                    activation=discriminator_block_activation
                )
            )

        self.final_conv = CoordConv(final_chan, 1, kernel_size = 2)

        self.add_layer_iters = add_layer_iters
        self.register_buffer('alpha', torch.tensor(0.))
        self.register_buffer('resolution', torch.tensor(init_resolution))
        self.register_buffer('iterations', torch.tensor(0.))

    def increase_resolution_(self):
        if self.resolution >= self.image_size:
            return

        self.alpha += self.alpha + (1 - self.alpha)
        self.iterations.fill_(0.)
        self.resolution *= 2

    def update_iter_(self):
        self.iterations += 1
        self.alpha -= (1 / self.add_layer_iters)
        self.alpha.clamp_(min = 0.)

    def forward(self, img):
        x = img

        for resolution, from_rgb, layer in zip(self.resolutions, self.from_rgb_layers, self.layers):
            if self.resolution < resolution:
                continue

            if self.resolution == resolution:
                x = from_rgb(x)

            if bool(resolution == (self.resolution // 2)) and bool(self.alpha > 0):
                x_down = F.interpolate(img, scale_factor = 0.5)
                x = x * (1 - self.alpha) + from_rgb(x_down) * self.alpha

            x = layer(x)

        out = self.final_conv(x)
        return out