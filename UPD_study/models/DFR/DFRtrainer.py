"""
adapted from https://github.com/YoungGod/DFR
"""
from argparse import ArgumentParser
from time import time
import numpy as np
import torch
from typing import Tuple
from torch import Tensor
from torch.nn import functional as F
from dfr_utils import estimate_latent_channels
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from DFRmodel import Extractor, FeatureAE, _set_requires_grad_false
from UPD_study.utilities.common_config import common_config
from UPD_study.utilities.evaluate import evaluate
from UPD_study.utilities.utils import (save_model, seed_everything,
                                       load_data, load_pretrained,
                                       misc_settings, ssim_map,
                                       load_model, log)

""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""


def get_config():
    parser = ArgumentParser()
    parser = common_config(parser)

    # Training Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--max_steps', '-ms', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--batch_size', '-bs', type=int, default=4, help='Batch size')

    # Model settings
    parser.add_argument('--arch', type=str, default='vgg19', choices=['vgg19', 'wide_resnet50_2'])
    parser.add_argument('--latent_channels', type=int, default=None, help='Number of CAE latent channels.')
    parser.add_argument('--start_layer', type=int, default=0, help='First backbone layer to use.')
    parser.add_argument('--last_layer', type=int, default=14, help='Last backbone layer to use.')

    # stride=2 outputs embedding volume: c x 64 x 64 for input 128
    # paper uses stride=4 for input 256 to get the same embedding volume size,  but for 128 it downsamples
    # too much and performs worse
    parser.add_argument('--stride', type=int, default=2, help='Stride of mean filter.')

    return parser.parse_args()


# set initial script settings
config = get_config()
config.method = 'DFR'
misc_settings(config)

""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""

# specific seed for creating the dataloader
seed_everything(42)

train_loader, val_loader, big_testloader, small_testloader = load_data(config)


""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""
# Reproducibility
seed_everything(config.seed)

if config.image_size == 256:
    config.stride = 4


# if config.modality == 'CXR':
#     config.latent_channels = 474
# if config.modality == 'MRI' and config.sequence == 't2':
#     config.latent_channels = 162
# if config.modality == 'MRI' and config.sequence == 't1':
#     config.latent_channels = 191

if config.latent_channels is None:
    print('Estimating number of required latent channels')

    extractor = Extractor(start_layer=config.start_layer,
                          last_layer=config.last_layer,
                          featmap_size=config.image_size,
                          stride=config.stride).to(config.device)

    config.latent_channels = estimate_latent_channels(extractor, train_loader)
    print('Estimated number of latent channels:{}'.format(config.latent_channels))
    del(extractor)


# Init model
print("Initializing model...")

model = FeatureAE(
    img_size=config.image_size,
    latent_channels=config.latent_channels,
    start_layer=config.start_layer,
    last_layer=config.last_layer,
    stride=config.stride
).to(config.device)


# load pretrained with CCD
if config.load_pretrained:
    config.modality = 'MRI' if config.modality == "MRInoram" else config.modality
    model.extractor.feat_extractor.backbone = load_pretrained(model.extractor.feat_extractor.backbone, config)
    _set_requires_grad_false(model.extractor.feat_extractor.backbone)
    if config.arch == 'vgg19':
        backbone_feat_modules = list(model.extractor.feat_extractor.backbone.features.children())
        model.extractor.feat_extractor.features = nn.Sequential(
            *(backbone_feat_modules + [model.extractor.feat_extractor.backbone.avgpool]))
        _set_requires_grad_false(model.extractor.feat_extractor.backbone)

# Init optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config.lr, weight_decay=config.weight_decay)

# Load saved model to evaluate
if config.eval:
    model.load_state_dict(load_model(config))
    print('Saved model loaded.')

# Space Benchmark
if config.space_benchmark:
    from torchinfo import summary
    a = summary(model, (16, 3, 128, 128), verbose=0)
    params = a.total_params
    print('Number of Million parameters: ', params / 1e06)
    exit(0)
""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""


def train_step(input) -> Tuple[float, Tensor]:
    model.train()
    optimizer.zero_grad()
    feats, rec = model(input)
    loss = torch.mean((feats - rec) ** 2)
    loss.backward()
    optimizer.step()
    return loss.item(), rec


def val_step(input, test_samples: bool = False) -> Tuple[float, Tensor, Tensor]:
    """Calculates val loss, anomaly maps of shape batch_shape and anomaly scores of shape [b,1]"""
    model.eval()

    with torch.no_grad():

        feats, rec = model(input)
        map_small = torch.mean((feats - rec) ** 2, dim=1, keepdim=True)
        loss = map_small.mean()

        if config.ssim_eval:
            anom_map_small = ssim_map(feats, rec)
            anomaly_map = F.interpolate(anom_map_small, input.shape[-2:], mode='bilinear', align_corners=True)

            if config.gaussian_blur:
                anomaly_map = anomaly_map.cpu().numpy()
                for i in range(anomaly_map.shape[0]):
                    anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)
                anomaly_map = torch.from_numpy(anomaly_map).to(config.device)

        else:
            anomaly_map = F.interpolate(map_small, input.shape[-2:], mode='bilinear', align_corners=True)
            if config.gaussian_blur:
                anomaly_map = anomaly_map.cpu().numpy()
                for i in range(anomaly_map.shape[0]):
                    anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)
                anomaly_map = torch.from_numpy(anomaly_map).to(config.device)

    if config.modality == 'MRI':
        mask = torch.stack([inp > inp.min() for inp in input])
        anomaly_map *= mask
        anomaly_score = torch.tensor([map[inp > inp.min()].max() for map, inp in zip(anomaly_map, input)])

    elif config.modality == 'RF':
        anomaly_score = torch.tensor([map.max() for map in anomaly_map])
    else:
        anomaly_score = torch.tensor([map.mean() for map in anomaly_map])

    if test_samples:
        return anomaly_map, anomaly_score
    else:
        return loss, anomaly_map, anomaly_score


def validate(val_loader, config):

    val_losses = []
    i_val_step = 0

    for input in val_loader:
        # x [b, 1, h, w]
        input = input.to(config.device)
        # Compute loss
        loss, anomaly_map, _ = val_step(input)
        val_losses.append(loss)
        i_val_step += 1

        if i_val_step >= config.val_steps:
            break

    # Print validation results
    log_msg = f"Validation loss on normal samples: {np.mean(val_losses):.4f}"
    print(log_msg)

    # Log to wandb
    log(
        {'val/loss': np.mean(val_losses)},
        config
    )

    log({
        'val/input': input,

        'val/res': anomaly_map,
    }, config)

    return np.mean(val_losses)


def train() -> None:

    print(f'Starting training {config.name}...')
    train_losses = []
    t_start = time()

    while True:
        for input in train_loader:
            config.step += 1
            input = input.to(config.device)
            loss, _ = train_step(input)

            # Add to losses
            train_losses.append(loss)

            if config.step % config.log_frequency == 0:

                # Print training loss
                log_msg = f"Iteration {config.step} - "
                log_msg += f"train loss: {np.mean(train_losses):.4f}"
                log_msg += f" - time: {time() - t_start:.2f}s"
                print(log_msg)

                # Log to wandb
                log({'train/loss': np.mean(train_losses)}, config)

                # Reset loss dict
                train_losses = []

            if config.step % config.val_frequency == 0:
                validate(val_loader, config)

            if config.step % config.anom_val_frequency == 0:
                evaluate(config, small_testloader, val_step)

            if config.step >= config.max_steps:
                save_model(model, config)
                print(f'Reached {config.max_steps} iterations. Finished training {config.name}.')
                return


if __name__ == '__main__':

    if config.eval:
        print('Evaluating model...')
        evaluate(config, big_testloader, val_step)

    else:
        train()
