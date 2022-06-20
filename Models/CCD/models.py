"""
adapted from: ccd
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models
import sys
sys.path.append('/data_ssd/users/lagi/thesis/UAD_study/')
from Models.VAE.VAEmodel import VAE
from Models.fanogan.GANmodel import Encoder


def backbone_architecture(config):
    if 'resnet' in config.backbone_arch:

        if config.backbone_arch == 'resnet18':
            backbone = models.resnet18(pretrained=True, progress=True)
            backbone.fc = nn.Identity()

            return {'backbone': backbone, 'dim': 512}

        elif config.backbone_arch == 'resnet50':
            backbone = models.resnet50(pretrained=True, progress=True)
            backbone.fc = nn.Identity()

            return {'backbone': backbone, 'dim': 2048}

        elif config.backbone_arch == 'wide_resnet50_2':
            backbone = models.wide_resnet50_2(pretrained=True, progress=True)
            backbone.fc = nn.Identity()

            return {'backbone': backbone, 'dim': 2048}

    elif 'vgg19' == config.backbone_arch:
        backbone = models.vgg19(pretrained=True, progress=True)
        return {'backbone': backbone, 'dim': 1000}

    elif 'vae' == config.backbone_arch:
        model = VAE(config)
        encoder = model.encoder
        bottleneck = model.bottleneck
        backbone = {'encoder': encoder, 'bottleneck': bottleneck}
        return {'backbone': backbone, 'dim': config.latent_dim * 2}

    elif 'fanogan' == config.backbone_arch:
        model = Encoder(config)
        backbone = model
        # 1024 instead of 128 of orig. architecture, with 128 ccd would not converge properly
        backbone.fc = nn.Linear(4 * 4 * 8 * 64, 1024)
        return {'backbone': backbone, 'dim': 1024}


class NormalizedLinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(NormalizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        with torch.no_grad():
            w = self.weight / self.weight.data.norm(keepdim=True, dim=0)
        return F.linear(x, w, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class ContrastiveModel(nn.Module):
    def __init__(self, config, features_dim=128):
        super(ContrastiveModel, self).__init__()

        backbone = backbone_architecture(config)
        self.backbone = backbone['backbone']

        # different handling for vae for practical purposes (saving)
        if config.backbone_arch == 'vae':
            self.is_vae = True
        else:
            self.is_vae = False

        self.backbone_dim = backbone['dim']
        self.class_num = 4  # num of strong aug versions
        self.cls_head_number = config.cls_head_number
        # in this implementation there's a more elegant way to do only contrastive head
        # https://github.com/sthalles/SimCLR/blob/master/run.py
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.backbone_dim, self.backbone_dim),
            nn.ReLU(),
            nn.Linear(self.backbone_dim, features_dim))

        if self.cls_head_number == 2:
            self.classification_head = nn.Sequential(nn.Linear(self.backbone_dim, features_dim),
                                                     nn.ReLU(),
                                                     NormalizedLinear(features_dim, self.class_num))
        # this doesn't work even though it was default in official implementation of paper
        else:
            self.classification_head = nn.Sequential(
                NormalizedLinear(self.backbone_dim, self.class_num))

    def forward(self, x):
        if self.is_vae:
            out = self.backbone['encoder'](x)
            out = self.backbone['bottleneck'](out)
        else:
            out = self.backbone(x)
        features = self.contrastive_head(out)
        features = F.normalize(features, dim=1)
        logits = self.classification_head(out)

        return features, logits
