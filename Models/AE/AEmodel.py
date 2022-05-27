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
    in_channels = config.img_channels

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


class AE(nn.Module):
    """
    A n-layer variational autoencoder
    adapted from: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
    """

    def __init__(self, config: Namespace):
        """
        config object should include kl_weight, kernel_size, padding, num_layers
        img_size, img_channels conv1x1, latent_dim
        """
        super().__init__()

        self.dropout = nn.Dropout(config.dropout)

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
            nn.Linear(intermediate_feats, config.latent_dim, bias=False),
        )

        self.decoder_input = nn.Sequential(
            nn.Linear(config.latent_dim, intermediate_feats,
                      bias=False),
            Reshape((-1, config.conv1x1, intermediate_res, intermediate_res)),
        )

        # Build decoder
        self.decoder = build_decoder(config)

    def loss_function(self, x: Tensor, y: Tensor) -> dict:

        loss = torch.mean((x - y) ** 2)

        return {
            'loss': loss
        }

    def forward(self, x: Tensor) -> Tensor:
        # https://doi.org/10.1145/1390156.1390294 Denoising Autoencoders
        x = self.dropout(x)
        # Encode
        res = self.encoder(x)
        decoder_inp = self.decoder_input(self.bottleneck(res))
        # Decode
        y = self.decoder(decoder_inp)
        return y
