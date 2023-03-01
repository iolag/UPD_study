# adapted from https://github.com/liuem607/expVAE
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, channel, height, width):
        super(Unflatten, self).__init__()
        self.channel = channel
        self.height = height
        self.width = width

    def forward(self, input):
        return input.view(input.size(0), self.channel, self.height, self.width)


class ConvVAE(nn.Module):

    def __init__(self, latent_size):
        super(ConvVAE, self).__init__()

        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(131072, 1024),
            nn.ReLU()
        )

        # hidden => mu
        self.fc1 = nn.Linear(1024, self.latent_size)

        # hidden => logvar
        self.fc2 = nn.Linear(1024, self.latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 131072),
            nn.ReLU(),
            Unflatten(128, 32, 32),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        return mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def reparameterize_eval(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, x: Tensor, y: Tensor, mu: Tensor, logvar: Tensor) -> dict:
        """
        Computes the VAE loss function.
        KL(N(mu, sigma), N(0, 1)) = log(frac{1}{sigma})+
         \frac{sigma^2 + mu^2}{2} - frac{1}{2}
        :param x: Input image
        :param y: Reconstructed image
        :param mu: Mean of the estimated latent Gaussian
        :param logvar: Standard deviation of the estimated latent Gaussian
        :param kl_weight: Weight of the KL divergence
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        recon_loss = F.binary_cross_entropy(y, x, reduction='mean')
        kl_loss = torch.mean(-0.5 * (1 + logvar - mu ** 2 - logvar.exp()))
        loss = recon_loss + 1 * kl_loss

        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
        }
