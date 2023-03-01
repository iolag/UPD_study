"""
adapted from: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
"""

import torch
import torch.nn as nn
from torch import Tensor
from argparse import Namespace


class Reshape(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self.size)


def build_encoder(config) -> nn.Module:

    # Build encoder

    hidden_dims = [config.width * 2 ** i for i in range(config.num_layers)]
    encoder = []
    in_channels = 3

    # Consecutive strided conv layers to downsample
    for h_dim in hidden_dims:
        encoder.append(
            nn.Sequential(
                nn.Conv2d(in_channels, h_dim, kernel_size=config.kernel_size, stride=2,
                          padding=config.padding),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(),
            )
        )
        in_channels = h_dim

    # Last layer conv 1x1 to reduce channel dimension to config.conv1x1
    encoder.append(nn.Sequential(
        nn.Conv2d(in_channels, config.conv1x1, kernel_size=1),
        nn.BatchNorm2d(config.conv1x1),
        nn.LeakyReLU(),
    ))

    return nn.Sequential(*encoder)


def build_decoder(config) -> nn.Module:
    # Build decoder
    hidden_dims = [config.width * 2 ** i for i in range(config.num_layers)]
    decoder = []

    # First layer 1x1 conv to expand channel dim
    decoder.append(nn.Sequential(
        nn.Conv2d(config.conv1x1, hidden_dims[-1], kernel_size=1),
        nn.BatchNorm2d(hidden_dims[-1]),
        nn.LeakyReLU(),
    ))

    # Consecutive transconvs(stride=2) to upscale img dim and decrease channel dim
    for i in range(len(hidden_dims) - 1, 0, -1):
        decoder.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i - 1],
                                   kernel_size=config.kernel_size, stride=2, padding=config.padding,
                                   output_padding=1),
                nn.BatchNorm2d(hidden_dims[i - 1]),
                nn.LeakyReLU(),
            )
        )
    # Last transconv doesn't decrease channel dim (Baur et al also does this)
    decoder.append(
        nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[0], hidden_dims[0],
                               kernel_size=config.kernel_size, stride=2, padding=config.padding,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.LeakyReLU(),
        )
    )

    # Final 1x1 conv layer to reduce channels to input channel dim
    decoder.append(
        nn.Conv2d(hidden_dims[0], config.img_channels, kernel_size=1)
    )

    return nn.Sequential(*decoder)


class VAE(nn.Module):

    def __init__(self, config: Namespace):
        """
        config object should include kl_weight, kernel_size, padding, num_layers
        img_size, img_channels conv1x1, latent_dim
        """
        super().__init__()

        self.dropout = nn.Dropout(config.dropout)

        self.kl_weight = config.kl_weight

        # find spatial resolution after last conv encoder layer using [(W−K+2P)/S]+1
        intermediate_res = config.image_size

        for i in range(config.num_layers):
            intermediate_res = (intermediate_res - config.kernel_size + 2 * config.padding) // 2 + 1

        # linear bottleneck input feature dimension
        intermediate_feats = intermediate_res * intermediate_res * config.conv1x1

        # Build encoder
        self.encoder = build_encoder(config)

        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(intermediate_feats, config.latent_dim * 2, bias=False),
        )

        self.decoder_input = nn.Sequential(
            nn.Linear(config.latent_dim, intermediate_feats,
                      bias=False),
            Reshape((-1, config.conv1x1, intermediate_res, intermediate_res)),
        )
        # Build decoder
        self.decoder = build_decoder(config)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu: Mean of the estimated latent Gaussian
        :param logvar: Standard deviation of the estimated latent Gaussian
        """
        unit_gaussian = torch.randn_like(mu)
        std = torch.exp(0.5 * logvar)
        return unit_gaussian * std + mu

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

        recon_loss = torch.mean((x - y) ** 2)
        kl_loss = torch.mean(-0.5 * (1 + logvar - mu ** 2 - logvar.exp()))
        loss = recon_loss + self.kl_weight * kl_loss

        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
        }

    def forward(self, x: Tensor) -> Tensor:
        # https://doi.org/10.1145/1390156.1390294
        x = self.dropout(x)

        # Grayscale case, required because CCD needs RGB input
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # Encode
        res = self.encoder(x)
        # Bottleneck
        mu, logvar = torch.chunk(self.bottleneck(res), 2, dim=1)
        # Reparametrize
        z = self.reparameterize(mu, logvar)
        # Decode
        decoder_inp = self.decoder_input(z)
        y = self.decoder(decoder_inp)

        return y, mu, logvar

    # # expVAE function
    # def reparameterize_eval(self, mu, logvar):
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return eps.mul(std).add_(mu)
