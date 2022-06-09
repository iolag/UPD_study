from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models


def _set_requires_grad_false(layer):
    for param in layer.parameters():
        param.requires_grad = False


class WR50FeatureExtractor(nn.Module):
    def __init__(
        self,
        start_layer: int = 1,  # layers from 1 to 4
        last_layer: int = 4
    ):
        """
        Returns features on multiple levels from a WideResnet50_2.

        Args:
            layer_names (list): List of string of layer names where to return
                                the features. Must be ordered
        """
        super().__init__()

        self.backbone = models.wide_resnet50_2(pretrained=True)
        # self.backbone.conv1.stride = 1
        self.outputs = []

        def hook(module, input, output):
            self.outputs.append(output)

        for i in range(start_layer, last_layer):
            if i == 0:
                self.backbone.layer1.register_forward_hook(hook)
            if i == 1:
                self.backbone.layer2.register_forward_hook(hook)
            if i == 2:
                self.backbone.layer3.register_forward_hook(hook)
            # if i == 3:
            #     self.backbone.layer3.register_forward_hook(hook)
            # if i == 4:
            #     self.backbone.layer4.register_forward_hook(hook)
        # self.print_dims(torch.ones((4, 3, 128, 128)))
        _set_requires_grad_false(self.backbone)

    def print_dims(self, inp: Tensor) -> None:
        with torch.no_grad():
            _ = self.backbone(inp)

        print('input size: [4,3,128,128]')
        for i, output in enumerate(self.outputs):
            print(f'backbone feature map {i} size: ', output.shape)

        self.outputs.clear()

    def forward(self, inp: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            inp (Tensor): Input tensor of shape [b, 1, h, w] or [b, 3, h, w]
        Returns:
            out (dict): Dictionary containing the extracted features as Tensors
        """
        if inp.shape[1] == 1:
            inp = inp.repeat(1, 3, 1, 1)

        with torch.no_grad():
            _ = self.backbone(inp)

        out = {}
        for i, output in enumerate(self.outputs):

            out[f'layer{i}'] = output

        self.outputs = []

        return out


class AvgFeatAGG2d(nn.Module):
    """
    Convolution operation with mean kernel.
    """

    def __init__(self, kernel_size: int, output_size: int = None,
                 dilation: int = 1, stride: int = 1):
        super(AvgFeatAGG2d, self).__init__()
        self.kernel_size = kernel_size
        # nn.Unfold : Extracts sliding local blocks from a batched input tensor.
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, stride=stride)
        # nn.Fold : Combines an array of sliding local blocks into a large containing tensor.
        self.fold = nn.Fold(output_size=output_size, kernel_size=1, dilation=1, stride=1)
        self.output_size = output_size

    def forward(self, input: Tensor) -> Tensor:
        """
        brings the input to shape (b, c, k*k , output_size*output_size),
        and takes the mean of dim = 2 to perform mean filtering
        """
        N, C, _, _ = input.shape

        output = self.unfold(input)  # (b, cxkxk, output_size*output_size)

        output = torch.reshape(output, (N, C, int(self.kernel_size**2),
                               int(self.output_size**2)))  # (b, c, k*k , output_size*output_size)

        output = torch.mean(output, dim=2)  # (b, c, output_size*output_size)

        output = self.fold(output)  # (b, c, output_size, output_size)

        return output


class Extractor(nn.Module):
    """
    Muti-scale regional feature based on VGG-feature maps.
    """

    def __init__(
        self,
        start_layer: int = 0,
        last_layer: int = 4,  # num of backbone layers to use
        kernel_size: int = 4,
        stride: int = 4,
        featmap_size: int = 128,  # input img size
        is_agg: bool = True,
    ):
        super().__init__()

        # instatiate VGG*FeatureExtractor class
        self.feat_extractor = WR50FeatureExtractor(start_layer, last_layer)

        self.featmap_size = featmap_size

        self.is_agg = is_agg

        # Calculate padding
        # not needed for stride=kernel_size=4, since it's 0 then and out_size can be calculated
        # but for other kernel,stride sizes, it is required for out_size calculation
        padding = (kernel_size - stride) // 2
        self.replicationpad = nn.ReplicationPad2d([padding] * 4)

        # Calculate output size, required to set up custom mean filter module
        self.out_size = int((featmap_size + 2 * padding - ((kernel_size - 1) + 1)) / stride + 1)

        if not self.is_agg:
            self.featmap_size = self.out_size

        # Find out how many channels we got from the backbone, needed for CAE input
        self.c_out = self.get_out_channels()

        self.feat_agg = AvgFeatAGG2d(kernel_size=kernel_size,
                                     output_size=self.out_size, stride=stride)

    def get_out_channels(self):
        '''
        calculates the sum of channels of every output feature map of the backbone
        (the num of channels of the output embedding volume)
        '''
        device = next(self.feat_extractor.parameters()).device
        inp = torch.randn((2, 1, 224, 224), device=device)
        feat_maps = self.feat_extractor(inp)
        channels = 0
        for feat_map in feat_maps.values():
            channels += feat_map.shape[1]
        return channels

    def forward(self, inp: Tensor) -> Tensor:
        feat_maps = self.feat_extractor(inp)

        features = []
        for _, feat_map in feat_maps.items():
            # Resizing to img_size
            feat_map = F.interpolate(feat_map, size=self.featmap_size,
                                     mode='bilinear',
                                     align_corners=True)

            # "aggregate" with 4x4 spatial mean filter
            # needs padding for the case of stride == 2 to make output_size 128 and not 127.
            if self.is_agg:
                feat_map = self.replicationpad(feat_map)
                feat_map = self.feat_agg(feat_map)

            features.append(feat_map)

        # Concatenate to tensor
        features = torch.cat(features, dim=1)

        return features


class FeatureAE(nn.Module):
    '''convolutional AE to model normal embedding volumes'''

    def __init__(self, img_size: int,
                 latent_channels: int,
                 use_batchnorm: bool = True,
                 start_layer: int = 1,
                 last_layer: int = 4,
                 stride: int = 4):

        super().__init__()

        torch.backends.cudnn.benchmark = False  # solves an error for specific latent channel nums

        self.extractor = Extractor(start_layer=start_layer,
                                   last_layer=last_layer,
                                   featmap_size=img_size,
                                   stride=stride)

        in_channels = self.extractor.c_out

        ks = 1
        pad = ks // 2

        # hidden channels of conv layers as in TABLE VII of paper
        hidden_channels = [
            in_channels,
            (in_channels + 2 * latent_channels) // 2,
            2 * latent_channels,
            latent_channels,
        ]

        # Encoder
        encoder = nn.Sequential()
        for i, (c_in, c_out) in enumerate(zip(hidden_channels[:-1], hidden_channels[1:])):
            encoder.add_module(f"encoder_conv_{i}",
                               nn.Conv2d(c_in, c_out, ks, padding=pad, bias=False))
            if use_batchnorm:
                encoder.add_module(f"encoder_batchnorm_{i}",
                                   nn.BatchNorm2d(c_out))
            encoder.add_module(f"encoder_relu_{i}", nn.ReLU())

        # Decoder
        decoder = nn.Sequential()
        for i, (c_in, c_out) in enumerate(zip(hidden_channels[-1::-1], hidden_channels[-2::-1])):
            decoder.add_module(f"decoder_conv_{i}",
                               nn.Conv2d(c_in, c_out, ks, padding=pad, bias=False))
            if use_batchnorm:
                decoder.add_module(f"decoder_batchnorm_{i}",
                                   nn.BatchNorm2d(c_out))
            decoder.add_module(f"decoder_relu_{i}", nn.ReLU())

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        feats = self.extractor(x)
        z = self.encoder(feats)
        rec = self.decoder(z)
        return feats, rec


if __name__ == '__main__':
    from argparse import Namespace
    torch.manual_seed(0)

    # Config
    config = Namespace()
    config.device = "cuda"
    config.img_size = 128
    config.latent_channels = 150

    stride = config.stride = 2  # stride 2 output embedding cx64x64 for input 128
    x = torch.randn((2, 1, config.img_size, config.img_size), device=config.device)

    # Create model

    ae = FeatureAE(img_size=config.img_size,
                   latent_channels=150,
                   use_batchnorm=True,
                   start_layer=1,
                   last_layer=4,
                   stride=2
                   ).to(config.device)

    # Sample input
    x = torch.randn((2, 1, config.img_size, config.img_size), device=config.device)

    # Forward
    feats, rec = ae(x)
    # print(feats.shape)
    print(x.shape, feats.shape, rec.shape)
