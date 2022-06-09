import sys
sys.path.append('/data_ssd/users/lagi/thesis/UAD_study/')
from argparse import ArgumentParser
from time import time
import numpy as np
import torch
from typing import Tuple
from torch import Tensor
from torch.nn import functional as F
from Models.DFR.dfr_utils import estimate_latent_channels
import wandb
import torch.nn as nn
from Utilities.common_config import common_config
from Utilities.evaluate import evaluate
from Utilities.utils import (save_model, seed_everything,
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


config = get_config()

config.method = 'DFR'

# general setup
misc_settings(config)

""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""

# specific seed for deterministic dataloader creation
seed_everything(42)

train_loader, val_loader, big_testloader, small_testloader = load_data(config)

# small_testloader = big_testloader

""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""
# Reproducibility
seed_everything(config.seed)

if config.image_size == 256:
    config.stride = 4

if config.arch == 'vgg19':
    from DFRmodel import Extractor, FeatureAE, _set_requires_grad_false
elif config.arch in ['wide_resnet50_2', 'resnet50']:
    from DFRmodelWR50 import Extractor, FeatureAE, _set_requires_grad_false


# this won't work with cfg.limited_metrics true
if config.latent_channels is None:
    print('Estimating number of required latent channels')

    extractor = Extractor(start_layer=config.start_layer,
                          last_layer=config.last_layer,
                          featmap_size=config.image_size,
                          stride=config.stride).to(config.device)

    # # load pretrained with CCD
    # if config.load_pretrained:
    #     config.modality = 'MRI' if config.modality == "MRInoram" else config.modality
    #     extractor.feat_extractor.backbone = load_pretrained(extractor.feat_extractor.backbone, config)
    #     _set_requires_grad_false(extractor.feat_extractor.backbone)
    #     if config.arch == 'vgg19':
    #         backbone_feat_modules = list(extractor.feat_extractor.backbone.features.children())
    #         extractor.feat_extractor.features = nn.Sequential(
    #             *(backbone_feat_modules + [extractor.feat_extractor.backbone.avgpool]))
    #         _set_requires_grad_false(extractor.feat_extractor.backbone)

    # if config.arch != 'vgg19':
    #     pass
    #     # extractor.feat_extractor.backbone.conv1.stride = 1

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

""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""
from scipy.ndimage import gaussian_filter
cos_loss = torch.nn.CosineSimilarity()


def loss_fucntion(feats, rec):
    # idx = [128, 256, 1024, 3072]
    idx = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    loss = 0
    start = 0
    for layer in idx:

        loss += torch.mean(1 - cos_loss(feats[start:layer].view(feats.shape[0], -1),
                                        rec[start:layer].view(rec.shape[0], -1)))
        start = layer
    return loss


def train_step(input) -> Tuple[float, Tensor]:
    model.train()
    optimizer.zero_grad()
    feats, rec = model(input)
    loss = loss_fucntion(feats, rec)
    loss.backward()
    optimizer.step()
    return loss.item(), rec


def get_anomaly_map(feats, rec, config) -> Tensor:
    # idx = [128, 256, 1024, 3072]
    idx = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512]

    anomaly_map = torch.zeros(feats.shape[0], 1, config.image_size, config.image_size).to(config.device)

    start = 0
    for layer in idx:
        fs = feats[:, start:layer]
        ft = rec[:, start:layer]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=config.image_size, mode='bilinear', align_corners=True)
        anomaly_map += a_map
        start = layer

    anomaly_map = anomaly_map.detach().cpu().numpy()
    # apply gaussian smoothing on the score map
    for i in range(anomaly_map.shape[0]):
        anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)

    anomaly_map = torch.from_numpy(anomaly_map).to(config.device)
    return anomaly_map


def val_step(input, return_loss: bool = True) -> Tuple[float, Tensor, Tensor]:
    """Calculates val loss, anomaly maps of shape batch_shape and anomaly scores of shape [b,1]"""
    model.eval()

    with torch.no_grad():

        feats, rec = model(input)
        map_small = torch.mean((feats - rec) ** 2, dim=1, keepdim=True)
        loss = map_small.mean()
        anomaly_map = get_anomaly_map(feats, rec, config)

    if config.modality == 'MRI':
        mask = torch.stack([inp > inp.min() for inp in input])
        anomaly_map *= mask
        anomaly_score = torch.tensor([map[inp > inp.min()].mean() for map, inp in zip(anomaly_map, input)])

    elif config.modality in ['RF']:
        mask = torch.stack([inp.mean(0, keepdim=True) > inp.mean(0, keepdim=True).min() for inp in input])
        anomaly_map *= mask
        anomaly_score = torch.tensor([map[inp.mean(0, keepdim=True) > inp.mean(0, keepdim=True).min()].mean()
                                     for map, inp in zip(anomaly_map, input)])
    else:
        anomaly_score = torch.tensor([map.mean() for map in anomaly_map])

    if return_loss:
        return loss.item(), anomaly_map, anomaly_score
    else:
        return anomaly_map, anomaly_score


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
    config.logger.log({
        'val/loss': np.mean(val_losses),
    }, step=config.step)

    # log images and anomaly maps
    input_images = list(input[:config.num_images_log].cpu())
    input_images = [wandb.Image(image) for image in input_images]

    anomaly_images = list(anomaly_map[:config.num_images_log].cpu())
    anomaly_images = [wandb.Image(image) for image in anomaly_images]

    config.logger.log({
        'val/input': input_images,
        'val/anom_image': anomaly_images,
    }, step=config.step)

    return np.mean(val_losses)


def train() -> None:

    print('Starting training...')
    i_epoch = 0
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

            # or config.step == 10 or config.step == 50 or config.step == 100 or config.step == 500:
            if config.step % config.anom_val_frequency == 0:

                evaluate(config, small_testloader, val_step, val_loader)

            if config.step % config.save_frequency == 0:
                save_model(model, config, config.step)

            if config.step >= config.max_steps:
                save_model(model, config)
                print(f'Reached {config.max_steps} iterations. Finished training {config.name}.')
                return

        i_epoch += 1


if __name__ == '__main__':

    if config.eval:
        print('Evaluating model...')
        evaluate(config, big_testloader, val_step, val_loader)

    else:
        train()
