import math
import torch
from torch import nn
from resnet import resnet18, resnet50, wide_resnet50_2
# FrEIA (https://github.com/VLL-HD/FrEIA/)
import FrEIA.framework as Ff
import FrEIA.modules as Fm


def positionalencoding2d(D, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 2 * dims_in),
                         nn.ReLU(),
                         nn.Linear(2 * dims_in, dims_out))


# freia_cflow
def load_decoder(config, dim_in):
    n_cond = config.condition_vec
    decoder = Ff.SequenceINN(dim_in)
    print('CNF coder:', dim_in)
    for k in range(config.coupling_blocks):
        decoder.append(Fm.AllInOneBlock,
                       cond=0,
                       cond_shape=(n_cond,),
                       subnet_constructor=subnet_fc,
                       affine_clamping=config.clamp_alpha,
                       global_affine_type='SOFTPLUS',
                       permute_soft=False)
    return decoder


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def load_encoder(config):
    """
    Loads encoder pretrained on natural images and forward hooks activations.

    Returns:
        encoder: Encoder model
        pool_layers: name list of pool layers eg. ['layer 1', 'layer 2' , 'layer 3']
        pool_dims: channel dimension of pool layers eg. [512, 1024, 2048]

    """
    pool_cnt = 0
    pool_dims = list()
    pool_layers = ['layer' + str(i) for i in range(config.num_pool_layers)]

    # load model

    if config.arch == 'resnet18':
        encoder = resnet18(pretrained=True, progress=True)
    elif config.arch == 'resnet50':
        encoder = resnet50(pretrained=True, progress=True)
    elif config.arch == 'wide_resnet50_2':
        encoder = wide_resnet50_2(pretrained=True, progress=True)

    if config.modality in ['MRI', 'CT']:

        # forward hook activations --> every forward pass, activations will be aggregated in activation dict
        if config.num_pool_layers >= 3:
            encoder.layer1.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in config.arch:
                pool_dims.append(encoder.layer1[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer1[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
        if config.num_pool_layers >= 2:
            encoder.layer2.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in config.arch:
                pool_dims.append(encoder.layer2[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer2[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
        if config.num_pool_layers >= 1:
            encoder.layer3.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in config.arch:
                pool_dims.append(encoder.layer3[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer3[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
    else:
        # forward hook activations --> every forward pass, activations will be aggregated in activation dict
        if config.num_pool_layers >= 3:
            encoder.layer2.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in config.arch:
                pool_dims.append(encoder.layer2[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer2[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
        if config.num_pool_layers >= 2:
            encoder.layer3.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in config.arch:
                pool_dims.append(encoder.layer3[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer3[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1
        if config.num_pool_layers >= 1:
            encoder.layer4.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in config.arch:
                pool_dims.append(encoder.layer4[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer4[-1].conv2.out_channels)
            pool_cnt = pool_cnt + 1

    return encoder, pool_layers, pool_dims
