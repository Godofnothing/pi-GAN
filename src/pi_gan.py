import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from torchvision.utils import save_image

import os

import pytorch_lightning as pl

from collections import OrderedDict

from functools import partial

from layers import Discriminator, Generator
from utils import gradient_penalty, get_item, log_loss, discriminator_loss, generator_loss

class piGAN(pl.LightningModule):
    def __init__(
        self,
        image_size,
        input_features,
        hidden_features,
        optim_cfg,
        generator_cfg, 
        discriminator_cfg,
        image_dataset,
        output_dir= "../generated_images",
        add_layers_iters: int = 10000,
        sample_every: int = 100,
        num_samples: int = 4,
        log_every: int = 10,
        gp_every : int = 4,
        batch_size: int = 32,
        loss_mode: str = "relu"
    ):
        super(piGAN, self).__init__()
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.optim_cfg = optim_cfg
        self.batch_size = batch_size
        self.image_dataset = image_dataset
        self.log_every = log_every
        self.gp_every = gp_every
        self.sample_every = sample_every
        self.add_layers_iters = add_layers_iters
        self.output_dir = output_dir
        self.num_samples = num_samples

        # in case the directory for generated images doesn't exist - create it
        os.makedirs(output_dir, exist_ok=True)

        self.G = Generator(
            image_size=image_size,
            input_features=input_features,
            hidden_features=hidden_features,
            **generator_cfg
        )

        self.D = Discriminator(
            image_size=image_size,
            **discriminator_cfg
        )

        # setup initial resolution for loaded_images
        self.image_dataset.set_transforms(self.D.init_resolution)

        self.iterations = 0
        self.last_loss_D = 0
        self.last_loss_G = 0

        self.discriminator_loss = partial(discriminator_loss, mode=loss_mode)
        self.generator_loss = partial(generator_loss, mode=loss_mode)

    def configure_optimizers(self):
        lr_discr = self.optim_cfg["discriminator"]["learning_rate"]
        target_lr_discr = self.optim_cfg["discriminator"]["target_learning_rate"]

        lr_gen = self.optim_cfg["generator"]["learning_rate"]
        target_lr_gen = self.optim_cfg["generator"]["target_learning_rate"]

        lr_decay_span = self.optim_cfg["learning_rate_decay_span"]
        
        self.optim_D = Adam(self.D.parameters(), betas=(0, 0.9), lr=lr_discr)
        self.optim_G = Adam(self.G.parameters(), betas=(0, 0.9), lr=lr_gen)

        D_decay_fn = lambda i: max(1 - i / lr_decay_span, 0) + (target_lr_discr / lr_discr) * min(i / lr_decay_span, 1)
        G_decay_fn = lambda i: max(1 - i / lr_decay_span, 0) + (target_lr_gen / lr_gen) * min(i / lr_decay_span, 1)

        self.sched_D = LambdaLR(self.optim_D, D_decay_fn)
        self.sched_G = LambdaLR(self.optim_G, G_decay_fn)

        return [self.optim_D, self.optim_G], [self.sched_D, self.sched_G]

    def forward(self, x):
        return self.G(x)

    def generate_samples(self, num_samples: int):
        rand_latents = torch.randn(num_samples, self.input_features)
        rand_latents = rand_latents.to(self.device)
        return self.forward(rand_latents)

    def training_step(self, batch, batch_idx, optimizer_idx):
        images = batch

        # gp
        apply_gp = self.iterations % self.gp_every == 0

        # train discriminator
        if optimizer_idx == 0:
            images = images.requires_grad_()

            # increase resolution
            if self.iterations % self.add_layers_iters == 0:
                if self.iterations != 0:
                    self.D.increase_resolution_()

                image_size = self.D.resolution.item()
                self.G.set_image_size(image_size)
                self.image_dataset.set_transforms(image_size)

            real_out = self.D(images)

            fake_images = self.generate_samples(self.batch_size)
            fake_out = self.D(fake_images.clone().detach())

            loss_D = self.discriminator_loss(fake_out, real_out)

            if apply_gp:
                gp = gradient_penalty(images, real_out)
                self.last_loss_gp = get_item(gp)
                loss = loss_D + gp
            else:
                loss = loss_D

            self.last_loss_D = loss_D

            tqdm_dict = {'loss_D': loss_D}
            output = OrderedDict({
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train generator
        if optimizer_idx == 1:
            fake_images = self.generate_samples(self.batch_size)
            fake_out = self.D(fake_images)

            loss_G = self.generator_loss(fake_out)

            self.last_loss_G = loss_G

            tqdm_dict = {'loss_G': loss_G}
            output = OrderedDict({
                'loss': loss_G,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })

            return output

    def training_epoch_end(self, outputs):
        self.D.update_iter_()
        self.iterations += 1

        if self.iterations % self.sample_every == 0:
            imgs = self.generate_samples(self.num_samples)
            imgs.clamp_(0., 1.)
            save_image(imgs, f'{self.output_dir}/generated_image_{self.iterations}.png', nrow = 2)
