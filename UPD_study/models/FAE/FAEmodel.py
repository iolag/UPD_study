"""

MIT License

Copyright (c) 2021 Felix Meissen

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
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models as tv_models
from pytorch_ssim import SSIMLoss


""""""""""""""""""""""" Feature Extractor """""""""""""""""""""""""""


RESNETLAYERS = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']


def _set_requires_grad_false(layer):
    for param in layer.parameters():
        param.requires_grad = False


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, resnet, layer_names: List[str] = RESNETLAYERS):
        """
        Returns features on multiple levels from a ResNet18.
        Available layers: 'layer0', 'layer1', 'layer2', 'layer3', 'layer4',
                          'avgpool'
        Args:
            resnet (nn.Module): Type of resnet used
            layer_names (list): List of string of layer names where to return
                                the features. Must be ordered
            pretrained (bool): Whether to load pretrained weights
        Returns:
            out (dict): Dictionary containing the extracted features as
                        torch.tensors
        """
        super().__init__()

        _set_requires_grad_false(resnet)

        # [b, 3, 256, 256]
        self.layer0 = nn.Sequential(
            *list(resnet.children())[:4])  # [b, 64, 64, 64]
        self.layer1 = resnet.layer1  # [b, 64, 64, 64]
        self.layer2 = resnet.layer2  # [b, 128, 32, 32]
        self.layer3 = resnet.layer3  # [b, 256, 16, 16]
        self.layer4 = resnet.layer4  # [b, 512, 8, 8]
        self.avgpool = resnet.avgpool  # [b, 512, 1, 1]
        # print(resnet.conv1.state_dict().keys())
        # print(self.layer0[0].state_dict().keys())
        self.layer_names = layer_names

    def forward(self, inp: Tensor) -> Dict[str, Tensor]:
        if inp.shape[1] == 1:
            inp = inp.repeat(1, 3, 1, 1)

        out = {}
        for name, module in self._modules.items():
            inp = module(inp)
            if name in self.layer_names:
                out[name] = inp
            if name == self.layer_names[-1]:
                break
        return out


class ResNet18FeatureExtractor(ResNetFeatureExtractor):
    def __init__(self, layer_names: List[str] = RESNETLAYERS,
                 pretrained: bool = True):
        super().__init__(tv_models.resnet18(pretrained=pretrained), layer_names)


class Extractor(nn.Module):
    """
    Muti-scale regional feature based on VGG-feature maps.
    """

    def __init__(
        self,
        cnn_layers: List[str] = ['layer1', 'layer2'],
        upsample: str = 'bilinear',
        inp_size: int = 128,
        keep_feature_prop: float = 1.0,
        pretrained: bool = True,
    ):
        super().__init__()

        self.backbone = ResNet18FeatureExtractor(layer_names=cnn_layers,
                                                 pretrained=pretrained)
        self.inp_size = inp_size
        self.featmap_size = inp_size // 4
        self.upsample = upsample
        self.align_corners = True if upsample == "bilinear" else None

        # Find out how many channels we got from the backbone
        c_feats = self.get_out_channels()

        # Create mask to drop random features_channels
        self.register_buffer('feature_mask', torch.Tensor(
            c_feats).uniform_() < keep_feature_prop)
        self.c_feats = self.feature_mask.sum().item()

    def get_out_channels(self) -> int:
        """Get the number of channels of the output feature map"""
        device = next(self.backbone.parameters()).device
        x = torch.randn((2, 1, self.inp_size, self.inp_size), device=device)
        return sum([feat_map.shape[1] for feat_map in self.backbone(x).values()])

    def forward(self, inp: Tensor) -> Tensor:
        # Center input
        # inp = (inp - 0.5) * 2

        # Extract feature maps
        feat_maps = self.backbone(inp)

        features = []
        for feat_map in feat_maps.values():
            # Resizing
            feat_map = F.interpolate(feat_map, size=self.featmap_size,
                                     mode=self.upsample,
                                     align_corners=self.align_corners)
            features.append(feat_map)

        # Concatenate to tensor
        features = torch.cat(features, dim=1)

        # Drop out feature maps
        features = features[:, self.feature_mask]

        return features


""""""""""""""""""""""" Feature Autoencoder """""""""""""""""""""""""""


def vanilla_feature_encoder(in_channels: int, hidden_dims: List[int],
                            norm_layer: str = None, dropout: float = 0.0,
                            bias: bool = False):
    """
    Vanilla feature encoder.
    Args:
        in_channels (int): Number of input channels
        hidden_dims (List[int]): List of hidden channel dimensions
        norm_layer (str): Normalization layer to use
        dropout (float): Dropout rate
        bias (bool): Whether to use bias
    Returns:
        encoder (nn.Module): The encoder
    """
    ks = 5  # Kernel size
    pad = ks // 2  # Padding

    # Build encoder
    enc = nn.Sequential()
    for i in range(len(hidden_dims)):
        # Add a new layer
        layer = nn.Sequential()

        # Convolution
        layer.add_module(f"encoder_conv_{i}",
                         nn.Conv2d(in_channels, hidden_dims[i], ks, stride=2,
                                   padding=pad, bias=bias))

        # Normalization
        if norm_layer is not None:
            layer.add_module(f"encoder_norm_{i}",
                             eval(norm_layer)(hidden_dims[i]))

        # LeakyReLU
        layer.add_module(f"encoder_relu_{i}", nn.LeakyReLU())

        # Dropout
        if dropout > 0:
            layer.add_module(f"encoder_dropout_{i}", nn.Dropout2d(dropout))

        # Add the layer to the encoder
        enc.add_module(f"encoder_layer_{i}", layer)

        in_channels = hidden_dims[i]

    # Final layer
    enc.add_module("encoder_conv_final",
                   nn.Conv2d(in_channels, in_channels, ks, stride=1,
                             padding=pad, bias=bias))

    return enc


def vanilla_feature_decoder(out_channels: int, hidden_dims: List[int],
                            norm_layer: str = None, dropout: float = 0.0,
                            bias: bool = False):
    """
    Vanilla feature decoder.
    Args:
        out_channels (int): Number of output channels
        hidden_dims (List[int]): List of hidden channel dimensions
        norm_layer (str): Normalization layer to use
        dropout (float): Dropout rate
        bias (bool): Whether to use bias
    Returns:
        decoder (nn.Module): The decoder
    """
    ks = 6  # Kernel size
    pad = 2  # Padding

    hidden_dims = [out_channels] + hidden_dims

    # Build decoder
    dec = nn.Sequential()
    for i in range(len(hidden_dims) - 1, 0, -1):
        # Add a new layer
        layer = nn.Sequential()

        # Transposed convolution
        layer.add_module(f"decoder_tconv_{i}",
                         nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i - 1],
                                            ks, stride=2, padding=pad,
                                            bias=bias))

        # Normalization
        if norm_layer is not None:
            layer.add_module(f"decoder_norm_{i}",
                             eval(norm_layer)(hidden_dims[i - 1]))

        # LeakyReLU
        layer.add_module(f"decoder_relu_{i}", nn.LeakyReLU())

        # Dropout
        if dropout > 0:
            layer.add_module(f"decoder_dropout_{i}", nn.Dropout2d(dropout))

        # Add the layer to the decoder
        dec.add_module(f"decoder_layer_{i}", layer)

    # Final layer
    dec.add_module("decoder_conv_final",
                   nn.Conv2d(hidden_dims[0], out_channels, 1, bias=False))

    return dec


class FeatureAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.enc = vanilla_feature_encoder(config.in_channels,
                                           config.hidden_dims,
                                           norm_layer="nn.BatchNorm2d",
                                           dropout=config.dropout,
                                           bias=False)
        self.dec = vanilla_feature_decoder(config.in_channels,
                                           config.hidden_dims,
                                           norm_layer="nn.BatchNorm2d",
                                           dropout=config.dropout,
                                           bias=False)

    def forward(self, x: Tensor) -> Tensor:
        z = self.enc(x)
        rec = self.dec(z)
        return rec


class FeatureReconstructor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.extractor = Extractor(inp_size=config.image_size,
                                   cnn_layers=config.extractor_cnn_layers,
                                   keep_feature_prop=config.keep_feature_prop,
                                   pretrained=True)

        config.in_channels = self.extractor.c_feats
        self.ae = FeatureAutoencoder(config)

        if config.loss_fn == 'ssim':
            self.loss_fn = SSIMLoss(window_size=5, size_average=False)
        elif config.loss_fn == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Unknown loss function: {config.loss_fn}")

    def forward(self, x: Tensor):
        with torch.no_grad():
            feats = self.extractor(x)
        return feats, self.ae(feats)

    def get_feats(self, x: Tensor) -> Tensor:
        return self.extractor(x)

    def get_rec(self, feats: Tensor) -> Tensor:
        return self.ae(feats)

    def loss(self, x: Tensor):
        feats, rec = self(x)
        loss = self.loss_fn(rec, feats).mean()
        return {'rec_loss': loss}

    def predict_anomaly(self, x: Tensor):
        """Returns per image anomaly maps and anomaly scores"""
        # Extract features
        feats, rec = self(x)

        # Compute anomaly map
        anomaly_map = self.loss_fn(rec, feats).mean(1, keepdim=True)
        anomaly_map = F.interpolate(anomaly_map, x.shape[-2:], mode='bilinear',
                                    align_corners=True)

        return anomaly_map + 1
