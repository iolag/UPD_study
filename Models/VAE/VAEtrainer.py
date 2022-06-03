import sys
sys.path.append('/data_ssd/users/lagi/thesis/UAD_study/')
from argparse import ArgumentParser
import numpy as np
import torch
from collections import defaultdict
from VAEmodel import VAE
from time import time
import wandb
from torch import Tensor
from typing import Tuple
from Utilities.evaluate import evaluate  # eval_reconstruction_based
from Utilities.common_config import common_config
from Utilities.utils import (save_model, seed_everything,
                             load_data, load_pretrained,
                             misc_settings, ssim_map,
                             load_model)

""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""


def get_config():
    parser = ArgumentParser()
    parser = common_config(parser)

    # Training Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--max_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    # Model Hyperparameters
    parser.add_argument('--kl_weight', type=float, default=0.001, help='kl weight')
    parser.add_argument('--latent_dim', type=int, default=512, help='Model width')
    parser.add_argument('--num_layers', type=int, default=6, help='Model width')
    parser.add_argument('--width', type=int, default=16, help='First conv layer num of filters')
    parser.add_argument('--conv1x1', type=int, default=16,
                        help='Channel downsampling with 1x1 convs before bottleneck')
    parser.add_argument('--kernel_size', type=int, default=3, help='convolutional kernel size')
    parser.add_argument('--padding', type=int, default=1,
                        help='padding for consistent downsampling, set 2 if kernel_size == 5')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Input Dropout like https://doi.org/10.1145/1390156.1390294')

    return parser.parse_args()


config = get_config()

# get logger and naming string
config.method = 'VAE'
config.naming_str, logger = misc_settings(config)

""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""

# specific seed for deterministic dataloader creation
seed_everything(42)

if not config.eval or config.norm_fpr:
    train_loader, val_loader, big_testloader, small_testloader = load_data(config)
else:
    big_testloader, small_testloader = load_data(config)

""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""
# Reproducibility
seed_everything(config.seed)

print("Initializing model...")
model = VAE(config).to(config.device)

if config.load_pretrained and not config.eval:
    config.arch = 'vae'
    model.encoder = load_pretrained(model.encoder, config)

# Init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0., 0.9),
                             weight_decay=config.weight_decay)

# Load saved model to evaluate
if config.eval:
    model.load_state_dict(load_model(config))
    print('Saved model loaded.')

""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""


def vae_train_step(model, optimizer, input) -> dict:
    model.train()
    optimizer.zero_grad()
    input_recon, mu, logvar = model(input.repeat(1, 1, 1, 1))
    loss_dict = model.loss_function(input, input_recon, mu, logvar)  # VAE loss
    loss = loss_dict['loss']
    loss.backward()
    optimizer.step()
    return loss_dict


def vae_val_step(model, input, return_loss: bool = True) -> Tuple[dict, Tensor]:

    model.eval()
    with torch.no_grad():
        input_recon, mu, logvar = model(input.repeat(1, 1, 1, 1))
    loss_dict = model.loss_function(input, input_recon, mu, logvar)  # VAE Loss

    # Anomaly map
    if config.ssim_eval:
        anomaly_map = ssim_map(input_recon, input)
    else:
        anomaly_map = (input - input_recon).abs().mean(1, keepdim=True)
    # for MRI, apply brainmask
    if config.modality == 'MRI':
        mask = torch.stack([inp > inp.min() for inp in input])
        anomaly_map *= mask
        input_recon *= mask
    if config.modality == 'MRI':
        anomaly_score = torch.tensor([map[inp > inp.min()].mean() for map, inp in zip(anomaly_map, input)])
    else:
        anomaly_score = torch.tensor([map.mean() for map in anomaly_map])

    if return_loss:
        return loss_dict, anomaly_map, anomaly_score, input_recon
    else:
        return anomaly_map, anomaly_score, input_recon


def validate(model, val_loader, i_iter, config) -> None:

    val_losses = defaultdict(list)
    i_val_step = 0

    for input in val_loader:
        i_val_step += 1
        input = input.to(config.device)

        loss_dict, anomaly_map, anomaly_score, input_recon = vae_val_step(model, input)

        for k, v in loss_dict.items():
            val_losses[k].append(v.item())

        if i_val_step >= config.val_steps:
            break

    # Print validation results
    log_msg = 'Validation losses on normal samples: '
    log_msg += " - ".join([f'{k}: {np.mean(v):.4f}' for k, v in val_losses.items()])
    print(log_msg)

    # Log to wandb
    logger.log(
        {f'val/{k}': np.mean(v)
         for k, v in val_losses.items()},
        step=i_iter
    )

    # log images and residuals
    input_images = list(input.cpu()[:config.num_images_log])
    input_images = [wandb.Image(image) for image in input_images]

    reconstructions = list(input_recon.cpu()[:config.num_images_log])
    reconstructions = [wandb.Image(image) for image in reconstructions]

    residuals = list(anomaly_map[:config.num_images_log].cpu())
    residuals = [wandb.Image(image) for image in residuals]

    logger.log({
        'val/input': input_images,
        'val/recon': reconstructions,
        'val/res': residuals,
    }, step=i_iter)

    return np.mean(val_losses['loss'])


def train(model):

    print('Starting training VAE...')

    i_iter = 0
    i_epoch = 0
    train_losses = defaultdict(list)

    t_start = time()

    while True:
        for input in train_loader:
            i_iter += 1
            input = input.to(config.device)

            # Train step
            loss_dict = vae_train_step(model, optimizer, input)

            # Each step store train losses
            for k, v in loss_dict.items():
                train_losses[k].append(v.item())

            if i_iter % config.log_frequency == 0 or i_iter == 10 or i_iter == 50 or i_iter == 100:
                # Print training loss
                log_msg = " - ".join([f'{k}: {np.mean(v):.4f}' for k,
                                      v in train_losses.items()])
                log_msg = f"Iteration {i_iter} - " + log_msg
                log_msg += f" - time: {time() - t_start:.2f}s"
                print(log_msg)

                # Log to wandb
                logger.log(
                    {f'train/{k}': np.mean(v)
                     for k, v in train_losses.items()},
                    step=i_iter
                )

                # Reset loss dict
                train_losses = defaultdict(list)

            if i_iter % config.val_frequency == 0 or i_iter == 10 or i_iter == 50 or i_iter == 100:
                validate(model, val_loader, i_iter, config)

            if i_iter % config.anom_val_frequency == 0 or i_iter == 10 or i_iter == 50 or i_iter == 100:
                evaluate(model, small_testloader, i_iter,
                         vae_val_step, logger, config, val_loader)

            if i_iter % config.save_frequency == 0 and i_iter != 0:
                save_model(model, config)

            if i_iter >= config.max_steps:
                save_model(model, config)
                print(f'Reached {config.max_steps} iterations. Finished training {config.naming_str}.')
                return

        i_epoch += 1
        print(f'Finished epoch {i_epoch}, ({i_iter} iterations)')


if __name__ == '__main__':
    if config.eval:
        config.num_images_log = 100
        print('Evaluating model...')
        evaluate(model, big_testloader, 0, vae_val_step, logger, config, val_loader)

    else:
        train(model)
