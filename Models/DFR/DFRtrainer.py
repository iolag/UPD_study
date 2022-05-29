
import sys
sys.path.append('/home/ioannis/lagi/thesis')
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
from Utilities.evaluate import eval_dfr_pii
from Utilities.utils import (save_model,
                             seed_everything,
                             load_data,
                             load_pretrained,
                             misc_settings,
                             ssim_map,
                             load_model)

""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""


def get_config():
    parser = ArgumentParser()
    parser = common_config(parser)

    # Training Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--max_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--batch_size', '-bs', type=int, default=4, help='Batch size')

    # Model settings
    parser.add_argument('--arch', type=str, default='vgg19', choices=['vgg19', 'wide_resnet50_2'])
    parser.add_argument('--latent_channels', type=int, default=None, help='Number of CAE latent channels.')
    parser.add_argument('--start_layer', type=int, default=0, help='First backbone layer to use.')
    parser.add_argument('--last_layer', type=int, default=14, help='Last backbone layer to use.')
    parser.add_argument('--upsample_mode', type=str, default='bilinear',
                        help='Alingment step interpolation mode.')
    # stride 2 output embedding volume: c x 64 x 64 for input 128
    # For default of paper 256-->s=4 -> 64x64, s=2 -> 128x128. But for our 128 input s=4 doesn't work
    parser.add_argument('--stride', type=int, default=4,
                        help='Stride of mean filter. 4 -> embedding spatial size 64, 2 -> 128.')

    return parser.parse_args()


config = get_config()


# msg = "num_images_log should be lower or equal to batch size"
# assert (config.batch_size >= config.num_images_log), msg

# Select training device
config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# get logger and naming string
config.method = 'DFR'
config.naming_str, logger = misc_settings(config)

""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""

# specific seed for deterministic dataloader creation
seed_everything(42)

if config.eval:
    config.batch_size = 100

if not config.eval:
    train_loader, val_loader, big_testloader, small_testloader = load_data(config)
else:
    big_testloader, small_testloader = load_data(config)

""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""
# Reproducibility
seed_everything(config.seed)

if config.arch == 'vgg19':
    from DFRmodel import Extractor, FeatureAE, _set_requires_grad_false
elif config.arch in ['wide_resnet50_2', 'resnet50']:
    from DFRmodelWR50 import Extractor, FeatureAE, _set_requires_grad_false

if config.latent_channels is None:
    print('Estimating number of required latent channels')

    extractor = Extractor(start_layer=config.start_layer,
                          last_layer=config.last_layer,
                          upsample_mode=config.upsample_mode,
                          featmap_size=config.image_size,
                          stride=config.stride).to(config.device)

    if config.arch != 'vgg19':
        extractor.feat_extractor.backbone.conv1.stride = 1

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
    upsample_mode=config.upsample_mode,
    stride=config.stride
).to(config.device)


# load pretrained with CCD
if config.load_pretrained:
    config.modality = 'MRI' if config.modality == "MRInoram" else config.modality
    model.extractor.feat_extractor.backbone = load_pretrained(model.extractor.feat_extractor.backbone, config)
    _set_requires_grad_false(model.extractor.feat_extractor.backbone)
    if config.arch == 'vgg19':
        backbone = list(model.extractor.feat_extractor.backbone.children())
        model.extractor.feat_extractor.features = nn.Sequential(
            *(backbone + [model.extractor.feat_extractor.backbone.avgpool]))

# Init optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config.lr, weight_decay=config.weight_decay)
# print
# model.extractor.feat_extractor.print_dims(torch.ones((4, 3, 128, 128)).cuda())

# Load saved model toevaluate
if config.eval:
    model = load_model(model, config)
    print('Saved model loaded.')

""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""


def train_step(model, optimizer, input) -> Tuple[float, Tensor]:
    model.train()
    optimizer.zero_grad()
    feats, rec = model(input)
    loss = torch.mean((feats - rec) ** 2)
    loss.backward()
    optimizer.step()
    return loss.item(), rec


def val_step(model, input, return_loss: bool = True) -> Tuple[float, Tensor, Tensor]:
    """Calculates val loss, anomaly maps of shape batch_shape and anomaly scores of shape [b,1]"""
    model.eval()

    with torch.no_grad():

        feats, rec = model(input)
        map_small = torch.mean((feats - rec) ** 2, dim=1, keepdim=True)
        loss = map_small.mean()

        anomaly_map = F.interpolate(map_small, input.shape[-2:], mode='bilinear', align_corners=True)

        if config.ssim_eval:
            anom_map_small = ssim_map(feats, rec)
            anomaly_map = F.interpolate(anom_map_small, input.shape[-2:], mode='bilinear', align_corners=True)
        else:
            anomaly_map = F.interpolate(map_small, input.shape[-2:], mode='bilinear', align_corners=True)

    if config.modality in ['MRI', 'MRInoram']:
        mask = torch.stack([inp > inp.min() for inp in input])
        anomaly_map *= mask

    anomaly_score = torch.tensor([map[inp > inp.min()].mean() for map, inp in zip(anomaly_map, input)])

    if return_loss:
        return loss.item(), anomaly_map, anomaly_score
    else:
        return anomaly_map, anomaly_score


def validate(model, val_loader, i_iter, config, logger):

    val_losses = []
    i_val_step = 0

    for input in val_loader:
        # x [b, 1, h, w]
        input = input.to(config.device)
        # Compute loss
        loss, anomaly_map, _ = val_step(model, input)
        val_losses.append(loss)
        i_val_step += 1

        if i_val_step >= config.val_steps:
            break

    # Print validation results
    log_msg = f"Validation loss on normal samples: {np.mean(val_losses):.4f}"
    print(log_msg)

    # Log to wandb
    logger.log({
        'val/loss': np.mean(val_losses),
    }, step=i_iter)

    # log images and anomaly maps
    input_images = list(input[:config.num_images_log].cpu())
    input_images = [wandb.Image(image) for image in input_images]

    anomaly_images = list(anomaly_map[:config.num_images_log].cpu())
    anomaly_images = [wandb.Image(image) for image in anomaly_images]

    logger.log({
        'val/input': input_images,
        'val/anom_image': anomaly_images,
    }, step=i_iter)

    return np.mean(val_losses)


def train(model, optimizer, train_loader, val_loader, small_testloader, config) -> None:

    print('Starting training...')
    i_iter = 0
    i_epoch = 0
    train_losses = []
    t_start = time()

    while True:
        for input in train_loader:
            i_iter += 1
            input = input.to(config.device)
            loss, _ = train_step(model, optimizer, input)

            # Add to losses
            train_losses.append(loss)

            if i_iter % config.log_frequency == 0:

                # Print training loss
                log_msg = f"Iteration {i_iter} - "
                log_msg += f"train loss: {np.mean(train_losses):.4f}"
                log_msg += f" - time: {time() - t_start:.2f}s"
                print(log_msg)

                # Log to wandb

                logger.log({
                    'train/loss': np.mean(train_losses),
                }, step=i_iter)

                # Reset loss dict
                train_losses = []

            if i_iter % config.val_frequency == 0:
                validate(model, val_loader, i_iter, config, logger)

            if i_iter % config.anom_val_frequency == 0 or i_iter == 10 or i_iter == 50 or i_iter == 100:
                eval_dfr_pii(model, small_testloader, i_iter, val_step, logger, config)

            if i_iter % config.save_frequency == 0 and i_iter != 0:
                save_model(model, config)

            if i_iter >= config.max_steps:
                save_model(model, config)
                print(f'Reached {config.max_steps} iterations. Finished training {config.naming_str}.')
                return

        i_epoch += 1


if __name__ == '__main__':

    if config.eval:
        config.num_images_log = 100
        print('Evaluating model...')
        eval_dfr_pii(model, big_testloader, 0, val_step, logger, config)

    else:
        train(model, optimizer, train_loader, val_loader, small_testloader, config)
