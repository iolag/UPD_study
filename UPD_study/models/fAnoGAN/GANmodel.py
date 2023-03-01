'''
adapted from https://github.com/dbbbbm/f-AnoGAN-PyTorch
by Felix Meissen https://github.com/FeliMe/
'''
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, autograd


DIM = 64
LATENT_DIM = 128

# works only for (128,128) input images, requires
# modifications for alternative img dimensions


def calc_gradient_penalty(D: nn.Module, x_real: Tensor, x_fake: Tensor) -> Tensor:
    """
    Calculate the gradient penalty loss for WGAN-GP.
    Gradient norm for an interpolated version of x_real and x_fake.
    See https://arxiv.org/abs/1704.00028

    :param D: Discriminator
    :param x_real: Real images
    :param x_fake: Fake images
    """

    # Useful variables
    device = x_real.device
    b = x_real.size(0)

    # Interpolate images
    alpha = torch.rand(b, 1, 1, 1, device=device)
    interp = alpha * x_real.detach() + ((1 - alpha) * x_fake.detach())
    interp.requires_grad = True

    # Forward discrminator
    d_out, _ = D(interp)

    # Calculate gradients
    grads = autograd.grad(outputs=d_out, inputs=interp,
                          grad_outputs=torch.ones_like(d_out),
                          create_graph=True)[0]
    grads = grads.view(b, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()

    return gp


""" Helper functions """


def initialize(net: nn.Module):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):  # Also includes custom Conv2d
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)


""" Modules """


class Conv2d(nn.Conv2d):
    """nn.Conv2d with same padding"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, bias: bool = True) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding=kernel_size // 2, bias=bias)


class UpSampleConv(nn.Module):
    """arXiv:1609.05158"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 bias: bool = True):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, bias=bias)

    def forward(self, inp: Tensor) -> Tensor:
        out = inp

        # repeat channel dim 2**2 times
        out = out.repeat(1, 4, 1, 1)

        # rearrange [b, c * 2**2, h, w] to [b, c, h*2, w*2] (equiv. to tensorflow.nn.depth_to_space)
        out = F.pixel_shuffle(out, 2)

        # convolve to downsample channel dim
        out = self.conv(out)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 resample: str, hw: int = None,
                 norm_layer: str = "batchnorm"):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: size of the convolving kernel
        :param resample: 'down', or 'up'
        :param hw (int): height and width of the image, required when downsampling
        :param norm_layer: layer that will be used for normalization
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resample = resample

        # create up or downsampling ResBlock components
        if resample == 'down':
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2),
                Conv2d(in_channels, out_channels, kernel_size=1),
            )
            self.conv1 = Conv2d(in_channels, in_channels, kernel_size,
                                bias=False)
            self.conv2 = nn.Sequential(
                Conv2d(in_channels, out_channels, kernel_size=1),
                nn.AvgPool2d(2),
            )
            bn1_channels = in_channels
            bn2_channels = in_channels

        elif resample == 'up':
            self.shortcut = UpSampleConv(
                in_channels, out_channels, kernel_size=1)
            self.conv1 = UpSampleConv(
                in_channels, out_channels, kernel_size, bias=False)
            self.conv2 = Conv2d(out_channels, out_channels, kernel_size)
            bn1_channels = in_channels
            bn2_channels = out_channels

        else:
            raise RuntimeError('resample must be either "down" or "up"')

        if norm_layer == "batchnorm":
            self.bn1 = nn.BatchNorm2d(bn1_channels)
            self.bn2 = nn.BatchNorm2d(bn2_channels)
        elif norm_layer == "layernorm":
            self.bn1 = nn.LayerNorm([bn1_channels, hw, hw])
            self.bn2 = nn.LayerNorm([bn2_channels, hw, hw])
        else:
            raise RuntimeError('norm_layer must be either "batchnorm" or "layernorm"')

    def forward(self, inp: Tensor) -> Tensor:
        shortcut = self.shortcut(inp)

        # Layer 1
        out = self.bn1(inp)
        out = torch.relu(out)
        out = self.conv1(out)

        # Layer 2
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.conv2(out)

        return shortcut + out


""" Models """


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Unpack config
        dim = config.dim if 'dim' in config else DIM
        img_channels = config.img_channels if 'img_channels' in config else 1
        latent_dim = config.latent_dim if 'latent_dim' in config else LATENT_DIM

        self.latent_dim = latent_dim

        # Define layers
        self.ln1 = nn.Linear(latent_dim, 4 * 4 * 8 * dim)

        resblock = partial(ResidualBlock, resample='up', norm_layer="batchnorm")

        # For 128x128
        self.rb1 = resblock(8 * dim, 8 * dim, kernel_size=3)
        self.rb2 = resblock(8 * dim, 4 * dim, kernel_size=3)
        self.rb3 = resblock(4 * dim, 4 * dim, kernel_size=3)
        self.rb4 = resblock(4 * dim, 2 * dim, kernel_size=3)
        self.rb5 = resblock(2 * dim, 1 * dim, kernel_size=3)

        # For 64x64
        # self.rb1 = resblock(8 * dim, 8 * dim, kernel_size=3)
        # self.rb2 = resblock(8 * dim, 4 * dim, kernel_size=3)
        # self.rb3 = resblock(4 * dim, 2 * dim, kernel_size=3)
        # self.rb4 = resblock(2 * dim, 1 * dim, kernel_size=3)

        self.bn = nn.BatchNorm2d(dim)
        self.conv1 = Conv2d(dim, img_channels, kernel_size=3)

        # Initialize weights
        initialize(self)

    def sample_latent(self, batch_size: int) -> Tensor:
        device = next(self.parameters()).device
        return torch.randn(batch_size, self.latent_dim, device=device)

    def forward(self, z: Tensor = None, batch_size: int = None) -> Tensor:
        if z is None:
            z = self.sample_latent(batch_size)

        out = z  # noise with shape [b, latent_dim]
        out = self.ln1(out)  # [b, 4*4*8*dim]
        out = out.view(out.shape[0], -1, 4, 4)  # [b, 8xdim, 4, 4]
        out = self.rb1(out)  # [b, 8*dim, 8, 8]
        out = self.rb2(out)  # [b, 4*dim, 16, 16]
        out = self.rb3(out)  # [b, 4*dim, 32, 32]
        out = self.rb4(out)  # [b, 2*dim, 64, 64]
        out = self.rb5(out)  # [b, 1*dim, 128, 128]

        out = torch.relu(self.bn(out))  # [b, 1*dim, 128, 128]
        out = torch.sigmoid(self.conv1(out))  # [b, img_channels, 128, 128]

        return out


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Unpack config
        dim = config.dim if 'dim' in config else DIM
        img_channels = config.img_channels if 'img_channels' in config else 1
        img_size = config.img_size if 'img_size' in config else 128
        # img_size = config.img_size if 'img_size' in config else 64C

        # Define layers
        self.conv1 = Conv2d(img_channels, dim, kernel_size=3)

        resblock = partial(ResidualBlock, resample='down', norm_layer="layernorm")
        # For 128x128
        self.rb1 = resblock(1 * dim, 2 * dim, kernel_size=3, hw=img_size)
        self.rb2 = resblock(2 * dim, 4 * dim, kernel_size=3, hw=img_size // 2)
        self.rb3 = resblock(4 * dim, 4 * dim, kernel_size=3, hw=img_size // 4)
        self.rb4 = resblock(4 * dim, 8 * dim, kernel_size=3, hw=img_size // 8)
        self.rb5 = resblock(8 * dim, 8 * dim, kernel_size=3, hw=img_size // 16)

        # For 64x64
        # self.rb1 = resblock(1 * dim, 2 * dim, kernel_size=3, hw=img_size)
        # self.rb2 = resblock(2 * dim, 4 * dim, kernel_size=3, hw=img_size // 2)
        # self.rb3 = resblock(4 * dim, 8 * dim, kernel_size=3, hw=img_size // 4)
        # self.rb4 = resblock(8 * dim, 8 * dim, kernel_size=3, hw=img_size // 8)

        self.ln1 = nn.Linear(4 * 4 * 8 * dim, 1)

        # Initialize weights
        initialize(self)

    def extract_feature(self, inp: Tensor) -> Tensor:
        """downsample input image to noise dimension"""

        out = inp  # [b, img_channels, 128, 128]
        out = self.conv1(out)  # [b, 1*dim, 128, 128]
        out = self.rb1(out)  # [b, 2*dim, 64, 64]
        out = self.rb2(out)
        out = self.rb3(out)
        out = self.rb4(out)
        out = self.rb5(out)  # [b, 8*dim, 4, 4]
        out = out.view(out.shape[0], -1)  # [b, 8*dim*4*4] (dim after ln1 of G)
        return out

    def forward(self, inp: Tensor) -> Tensor:

        # donwsample to [b, 8*dim*4*4] (dim after ln1 of G)
        feats = self.extract_feature(inp)

        # pass through decision layer (outputs critic score, a single float)
        out = self.ln1(feats)  # [b, 1]

        out = out.view(-1)  # [b]
        return out, feats


class Encoder(nn.Module):
    """
    D's architecture, but with an extra FC layer to bring input to noise dimension.
    """

    def __init__(self, config):
        super().__init__()

        # Unpack config
        dim = config.dim if 'dim' in config else DIM
        img_channels = 3
        latent_dim = config.latent_dim if 'latent_dim' in config else LATENT_DIM
        dropout = config.dropout if 'dropout' in config else 0.

        # Define layers
        self.dropout = nn.Dropout(dropout)
        self.conv_in = nn.Conv2d(img_channels, dim, 3, 1, padding=1)

        resblock = partial(ResidualBlock, resample='down', norm_layer="batchnorm")
        self.res1 = resblock(1 * dim, 2 * dim, kernel_size=3)
        self.res2 = resblock(2 * dim, 4 * dim, kernel_size=3)
        self.res3 = resblock(4 * dim, 4 * dim, kernel_size=3)
        self.res4 = resblock(4 * dim, 8 * dim, kernel_size=3)
        self.res5 = resblock(8 * dim, 8 * dim, kernel_size=3)

        # For 64x64
        # self.res1 = resblock(1 * dim, 2 * dim, kernel_size=3)
        # self.res2 = resblock(2 * dim, 4 * dim, kernel_size=3)
        # self.res3 = resblock(4 * dim, 8 * dim, kernel_size=3)
        # self.res4 = resblock(8 * dim, 8 * dim, kernel_size=3)

        self.fc = nn.Linear(4 * 4 * 8 * dim, latent_dim)

        # Initialize weights
        initialize(self)

    def forward(self, inp: Tensor) -> Tensor:
        # Grayscale case, required because CCD needs RGB input
        if inp.shape[1] == 1:
            inp = inp.repeat(1, 3, 1, 1)
        out = inp  # [b, img_channels, 128, 128]
        out = self.dropout(out)  # Appendix A.1 of paper
        out = self.conv_in(out)  # [b, 1*dim, 128, 128]
        out = self.res1(out)  # [b, 2*dim, 64, 64]
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)  # [b, 8xdim, 4, 4]
        out = out.view(out.size(0), -1)  # [b, 4*4*8*dim]
        out = self.fc(out)  # [b, latent_dim] back to noise dimension
        return torch.tanh(out)  # Appendix A.2 of paper


class fAnoGAN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.G = Generator(config)
        self.D = Discriminator(config)
        self.E = Encoder(config)

    def forward(self, x: Tensor, feat_weight: float = 1.) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Return the anomaly map, anomaly score, and reconstruction for the
        given input.
        :param x: Input image
        :param feat_weight: Weight of the feature difference in the anoamly score
        """
        x_rec = self.G(self.E(x))
        f_x, f_x_rec = self.D.extract_feature(torch.cat((x, x_rec), dim=0)).chunk(2, 0)

        # Anomaly map is the residual of the input and the reconstructed image
        anomaly_map = (x - x_rec).abs().mean(1, keepdim=True)

        # Anomaly score
        img_diff = (x - x_rec).pow(2).mean((1, 2, 3))
        feat_diff = (f_x - f_x_rec).pow(2).mean((1))
        anomaly_score = img_diff + feat_weight * feat_diff

        return anomaly_map, anomaly_score, x_rec


if __name__ == '__main__':
    # Config
    from argparse import Namespace
    config = Namespace()

    # Models
    model = fAnoGAN(config)
    # print("Generator:", model.G, "\n")
    # print("Discriminator:", model.D, "\n")
    # print("Encoder:", model.E, "\n")

    # Data
    x = torch.randn(2, 1, 128, 128)
    z = torch.randn(2, 128)

    # Forward
    # anomaly_map, anomaly_score, x_rec = model(x)
    # print(anomaly_map.shape, anomaly_score.shape, x_rec.shape)
    x_gen = model.G(z)
    d_out, d_feats = model.D(x)
    # z_gen = model.E(x_gen)
    print(x_gen.shape, d_out.shape, d_feats.shape)  # z_gen.shape)
    # import IPython
    # IPython.embed()
    # exit(1)
