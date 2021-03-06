"""A model for image classification task"""
# pylint: disable=R0901, R0913, R0914, W0613, W0221

import logging
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision

from src.data.datamodules import CelebDataModule

logger = logging.getLogger(__name__)

LAMBDA_GP = 10

class WGANModule(pl.LightningModule):
    """PyTorch-Lightning module for classification"""

    def __init__(
        self,
        learning_rate=2e-5,
    ):
        super().__init__()
        self.automatic_optimization = False

        self.lr = learning_rate
        self.save_hyperparameters()
        self.writer = SummaryWriter("logs")
        self.epoch_idx = 0
        self.critic_iterations = 5

        self.generator = Generator()
        self.generator.apply(weights_init)
        self.critic = Discriminator()
        self.critic.apply(weights_init)

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        (opt_g, opt_d), _= self.optimizers()
        images_real, noise= batch

        for _ in range(self.critic_iterations):
            G_output_fake = self.generator(noise)
            critic_output_real = self.critic(images_real).view(-1)
            critic_output_fake = self.critic(G_output_fake).view(-1)
            gp = gradient_penalty(self.critic, images_real, G_output_fake, device="cuda")
            loss_critic = (-(torch.mean(critic_output_real) - torch.mean(critic_output_fake)) + LAMBDA_GP * gp)

            self.critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_d.step()

        gen_fake = self.critic(G_output_fake).view(-1)
        loss_gen = -torch.mean(gen_fake)
        self.generator.zero_grad()
        loss_gen.backward()
        opt_g.step()

        '''
        # train discriminator
        if optimizer_idx == 1:
            D_output_real = self.discriminator(images_real).view(-1)
            D_loss_real = F.binary_cross_entropy(D_output_real.float(), torch.ones_like(D_output_real).float())

            D_output_fake = self.discriminator(G_output_fake.detach()).view(-1)
            D_loss_fake = F.binary_cross_entropy(D_output_fake.float(), torch.zeros_like(D_output_real).float())

            D_loss = D_loss_real + D_loss_fake
            self.log("D_loss", D_loss, logger=True)
            return D_loss

        # train generator
        if optimizer_idx == 0:
            D_output = self.discriminator(G_output_fake).view(-1)
            G_loss = F.binary_cross_entropy(D_output.float(), torch.ones_like(D_output).float())
            self.log("G_loss", G_loss, logger=True)
            return G_loss
        '''

    def training_epoch_end(self, outputs):
        with torch.no_grad():
            output_fake = self.generator(torch.randn(32, 100, 1, 1, device="cuda"))
            grid = torchvision.utils.make_grid(output_fake, normalize=True)
            self.writer.add_image("Fake images", grid, global_step=self.epoch_idx)
            self.epoch_idx += 1

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.0, 0.9))
        opt_d = torch.optim.Adam(self.critic.parameters(), lr=self.lr, betas=(0.0, 0.9))
        return [opt_g, opt_d], []


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        generator_input = 100
        ngf = 64
        nc = 3
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(generator_input, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.model(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        ndf = 64
        nc = 3
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        return self.model(input)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def gradient_penalty(critic, real, fake, device="cuda"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def configure_trainer(
    save_dir: Path,
    trainer_args: Dict,
    early_stopping_args: Dict = None,
) -> Trainer:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    callbacks = []
    if early_stopping_args:
        callbacks.append(EarlyStopping(**early_stopping_args))

    return Trainer(
        **trainer_args,
        callbacks=callbacks,
        logger=TensorBoardLogger(
            save_dir,
            name="",
            version="",
            default_hp_metric=False,
        ),
    )


def train(
    data_args: Dict,
    model_args: Dict,
    trainer_args: Dict,
    early_stopping_args: Dict,
    output_dir: Path,
):
    model = WGANModule(
       **model_args
    )
    data = CelebDataModule(
        **data_args,
    )
    trainer = configure_trainer(
        output_dir, trainer_args, early_stopping_args
    )
    trainer.fit(model, datamodule=data)
    