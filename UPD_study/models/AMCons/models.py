"""
MIT License

Copyright (c) 2022 Julio Silva-Rodríguez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torchvision
import numpy as np

# ------------------
# RESIDUAL MODELS


class Resnet(torch.nn.Module):
    def __init__(self, in_channels, n_blocks=4):
        super(Resnet, self).__init__()
        self.n_blocks = n_blocks
        self.nfeats = 512 // (2**(4 - n_blocks))

        self.input = torch.nn.Conv2d(in_channels=in_channels, out_channels=64,
                                     kernel_size=(7, 7), stride=(2, 2),
                                     padding=(3, 3), bias=False)
        resnet18_model = torchvision.models.resnet18(pretrained=False)
        self.resnet = torch.nn.Sequential(
            *(list(resnet18_model.children())[i + 4] for i in range(0, self.n_blocks)))

        # placeholder for the gradients
        self.gradients = None

    def forward(self, x):
        x = self.input(x)
        F = []
        for iBlock in range(0, self.n_blocks):
            x = list(self.resnet.children())[iBlock](x)
            F.append(x)

        return x, F


class Encoder(torch.nn.Module):
    def __init__(self, fin=1, zdim=128, dense=False, variational=False, n_blocks=4, spatial_dim=7,
                 gap=False):
        super(Encoder, self).__init__()
        self.fin = fin
        self.zdim = zdim
        self.dense = dense
        self.n_blocks = n_blocks
        self.gap = gap
        self.variational = variational

        # 1) Feature extraction
        self.backbone = Resnet(in_channels=self.fin, n_blocks=self.n_blocks)
        # 2) Latent space (dense or spatial)
        if self.dense:  # dense
            if gap:
                if self.variational:
                    self.mu = torch.nn.Conv2d(self.backbone.nfeats, zdim, (1, 1))
                    self.log_var = torch.nn.Conv2d(self.backbone.nfeats, zdim, (1, 1))
                else:
                    self.z = torch.nn.Conv2d(self.backbone.nfeats, zdim, (1, 1))
            else:
                if self.variational:
                    self.mu = torch.nn.Linear(self.backbone.nfeats * spatial_dim**2, zdim)
                    self.log_var = torch.nn.Linear(self.backbone.nfeats * spatial_dim**2, zdim)
                else:
                    self.z = torch.nn.Linear(self.backbone.nfeats * spatial_dim ** 2, zdim)
        else:  # spatial
            if self.variational:
                self.mu = torch.nn.Conv2d(self.backbone.nfeats, zdim, (1, 1))
                self.log_var = torch.nn.Conv2d(self.backbone.nfeats, zdim, (1, 1))
            else:
                self.z = torch.nn.Conv2d(self.backbone.nfeats, zdim, (1, 1))
        # fc layer used for pre-training with ccd
        self.fc = torch.nn.Linear(self.backbone.nfeats * spatial_dim**2, 1024)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)

        sample = mu + (eps * std)  # sampling
        return sample

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # 1) Feature extraction
        x, allF = self.backbone(x)

        if self.dense and not self.gap:
            x = torch.nn.Flatten()(x)

        if self.dense and self.gap:
            x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
            x = x.view(x.size(0), self.backbone.nfeats, 1, 1)
        # x.shape = [bs, 32768] with default hparams
        # 2) Latent space
        if self.variational:
            # get `mu` and `log_var`
            z_mu = self.mu(x)
            z_logvar = self.log_var(x)
            # get the latent vector through reparameterization
            z = self.reparameterize(z_mu, z_logvar)
        else:
            z = self.z(x)
            z_mu, z_logvar = None, None

        return z, z_mu, z_logvar, allF

    def forward_CCD(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # 1) Feature extraction
        x, allF = self.backbone(x)

        if self.dense and not self.gap:
            x = torch.nn.Flatten()(x)

        if self.dense and self.gap:
            x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
            x = x.view(x.size(0), self.backbone.nfeats, 1, 1)

        x = self.fc(x)
        return x


class Decoder(torch.nn.Module):

    def __init__(self, fin=256, nf0=128, n_channels=1, dense=False, n_blocks=4, spatial_dim=7,
                 gap=False):
        super(Decoder, self).__init__()
        self.n_blocks = n_blocks
        self.dense = dense
        self.spatial_dim = spatial_dim
        self.fin = fin
        self.gap = gap
        self.nf0 = nf0

        if self.dense and not self.gap:
            self.dense_layer = torch.nn.Sequential(torch.nn.Linear(fin, nf0 * spatial_dim**2))
        if not dense:
            self.dense_layer = torch.nn.Sequential(torch.nn.Conv2d(fin, nf0, (1, 1)))

        # Set number of input and output channels
        n_filters_in = [nf0 // 2**(i) for i in range(0, self.n_blocks + 1)]
        n_filters_out = [nf0 // 2**(i + 1) for i in range(0, self.n_blocks)] + [n_channels]

        self.blocks = torch.nn.ModuleList()
        for i in np.arange(0, self.n_blocks):
            self.blocks.append(torch.nn.Sequential(BasicBlock(n_filters_in[i], n_filters_out[i],
                                                              downsample=True),
                                                   BasicBlock(n_filters_out[i], n_filters_out[i])))
        self.out = torch.nn.Conv2d(n_filters_in[-1], n_filters_out[-1], kernel_size=(3, 3), padding=(1, 1))

        self.n_filters_in = n_filters_in
        self.n_filters_out = n_filters_out

    def forward(self, x):

        if self.dense and not self.gap:
            x = self.dense_layer(x)
            x = torch.nn.Unflatten(-1, (self.nf0, self.spatial_dim, self.spatial_dim))(x)

        if self.dense and self.gap:
            x = torch.nn.functional.interpolate(x, scale_factor=self.spatial_dim)

        if not self.dense:
            x = self.dense_layer(x)

        for i in np.arange(0, self.n_blocks):
            x = self.blocks[i](x)
        f = x
        out = self.out(f)

        return out, f


class ResBlock(torch.nn.Module):

    def __init__(self, fin, fout):
        super(ResBlock, self).__init__()
        self.conv_straight_1 = torch.nn.Conv2d(fin, fout, kernel_size=(3, 3), padding=(1, 1))
        self.bn_1 = torch.nn.BatchNorm2d(fout)
        self.conv_straight_2 = torch.nn.Conv2d(fout, fout, kernel_size=(3, 3), padding=(1, 1))
        self.bn_2 = torch.nn.BatchNorm2d(fout)
        self.conv_skip = torch.nn.Conv2d(fin, fout, kernel_size=(3, 3), padding=(1, 1))
        self.upsampling = torch.nn.Upsample(scale_factor=(2, 2))
        self.relu = torch.nn.ReLU()

    def forward(self, x):

        x_st = self.upsampling(x)
        x_st = self.conv_straight_1(x_st)
        x_st = self.relu(x_st)
        x_st = self.bn_1(x_st)
        x_st = self.conv_straight_2(x_st)
        x_st = self.relu(x_st)
        x_st = self.bn_2(x_st)

        x_sk = self.upsampling(x)
        x_sk = self.conv_skip(x_sk)

        out = x_sk + x_st

        return out


class BasicBlock(torch.nn.Module):

    def __init__(self, inplanes=32, planes=64, stride=1, downsample=False, bn=True):
        super().__init__()
        norm_layer = torch.nn.BatchNorm2d
        self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = norm_layer(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        if self.downsample:
            self.downsample_layer_conv = torch.nn.Sequential(
                torch.nn.Conv2d(inplanes, planes, kernel_size=(1, 1)))
            self.downsample_layer = torch.nn.Upsample(scale_factor=(2, 2))
            self.downsample_layer_bn = norm_layer(planes)
        self.stride = stride
        self.bn = bn

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.downsample:
            out = self.downsample_layer(out)
        if self.bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)

        if self.downsample:
            identity = self.downsample_layer_conv(identity)
            identity = self.downsample_layer(identity)
            if self.bn:
                identity = self.downsample_layer_bn(identity)

        out += identity
        out = self.relu(out)

        return out


class ConvBlock(torch.nn.Module):

    def __init__(self, fin, fout):
        super(ConvBlock, self).__init__()

        self.conv = torch.nn.Conv2d(fin, fout, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(fout)
        self.act = torch.nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):

        out = self.act(self.bn(self.conv(x)))

        return out


class DeconvBlock(torch.nn.Module):

    def __init__(self, fin, fout):
        super(DeconvBlock, self).__init__()

        self.conv = torch.nn.Conv2d(fin, fout, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(fout)
        self.act = torch.nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):

        out = self.act(self.bn(self.conv(x)))

        return out
