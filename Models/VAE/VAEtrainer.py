import sys
sys.path.append('/data_ssd/users/lagi/thesis/UAD_study/')
from argparse import ArgumentParser
import numpy as np
import torch
from collections import defaultdict
from VAEmodel import VAE
from time import time
from torch import Tensor
from typing import Tuple
from Utilities.evaluate import evaluate  # eval_reconstruction_based
from Utilities.common_config import common_config
from Utilities.utils import (save_model, seed_everything,
                             load_data, load_pretrained,
                             misc_settings, ssim_map,
                             load_model, log)

""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""


def get_config():
    parser = ArgumentParser()
    parser = common_config(parser)

    # Training Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--max_steps', type=int, default=30000, help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

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

# Specific modality params (Default above are for MRI t2)
if config.modality == 'CXR':
    config.kl_weight = 0.0001
    config.num_layers = 6
    config.latent_dim = 256
    config.width = 16
    config.conv1x1 = 64
    config.stadardize = True

if config.modality == 'COL':
    config.kl_weight = 0.0001
    config.num_layers = 6
    config.latent_dim = 256
    config.width = 16
    config.conv1x1 = 64
    config.stadardize = True

if config.modality == 'MRI' and config.sequence == 't1':
    config.kl_weight = 0.0001
    config.num_layers = 6
    config.latent_dim = 512
    config.width = 32
    config.conv1x1 = 64


# get logger and naming string
config.method = 'VAE'
misc_settings(config)

""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""

# specific seed for deterministic dataloader creation
seed_everything(42)

train_loader, val_loader, big_testloader, small_testloader = load_data(config)

""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""
# Reproducibility
seed_everything(config.seed)

print("Initializing model...")
model = VAE(config).to(config.device)

if config.load_pretrained and not config.eval:
    config.arch = 'vae'
    model = load_pretrained(model, config)

# Init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0., 0.9),
                             weight_decay=config.weight_decay)

# Load saved model to evaluate
if config.eval:
    model.load_state_dict(load_model(config))
    print('Saved model loaded.')

""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""


def vae_train_step(input) -> dict:
    model.train()
    optimizer.zero_grad()
    input_recon, mu, logvar = model(input)
    loss_dict = model.loss_function(input, input_recon, mu, logvar)  # VAE loss
    loss = loss_dict['loss']
    loss.backward()
    optimizer.step()
    return loss_dict


def vae_val_step(input, return_loss: bool = True) -> Tuple[dict, Tensor]:

    model.eval()

    with torch.no_grad():
        input_recon, mu, logvar = model(input)

    loss_dict = model.loss_function(input, input_recon, mu, logvar)  # VAE Loss

    # Anomaly map
    if config.ssim_eval:
        anomaly_map = ssim_map(input_recon, input)
    else:
        anomaly_map = (input - input_recon).abs().mean(1, keepdim=True)
    # for MRI, RF apply brainmask
    if config.modality == 'MRI':
        mask = torch.stack([inp > inp.min() for inp in input])
        anomaly_map *= mask
        input_recon *= mask
        anomaly_score = torch.tensor([map[inp > inp.min()].mean() for map, inp in zip(anomaly_map, input)])

    elif config.modality in ['RF']:
        mask = torch.stack([inp.mean(0, keepdim=True) > inp.mean(0, keepdim=True).min() for inp in input])
        anomaly_map *= mask
        anomaly_score = torch.tensor([map[inp.mean(0, keepdim=True) > inp.mean(0, keepdim=True).min()].mean()
                                     for map, inp in zip(anomaly_map, input)])
    else:
        anomaly_score = torch.tensor([map.mean() for map in anomaly_map])

    if return_loss:
        return loss_dict, anomaly_map, anomaly_score, input_recon
    else:
        return anomaly_map, anomaly_score, input_recon


def validate(val_loader, config) -> None:

    val_losses = defaultdict(list)
    i_val_step = 0

    for input in val_loader:

        i_val_step += 1

        input = input.to(config.device)

        loss_dict, anomaly_map, anomaly_score, input_recon = vae_val_step(input)

        for k, v in loss_dict.items():
            val_losses[k].append(v.item())

        if i_val_step >= config.val_steps:
            break

    # Print validation results
    log_msg = 'Validation losses on normal samples: '
    log_msg += " - ".join([f'{k}: {np.mean(v):.4f}' for k, v in val_losses.items()])
    print(log_msg)

    # Log to wandb
    log(
        {f'val/{k}': np.mean(v)
         for k, v in val_losses.items()},
        config
    )

    # log images and residuals

    log({
        'val/input': input,
        'val/recon': input_recon,
        'val/res': anomaly_map,
    }, config)

    return np.mean(val_losses['loss'])


def train(model):

    print('Starting training VAE...')
    i_epoch = 0
    train_losses = defaultdict(list)
    t_start = time()

    while True:

        for input in train_loader:

            config.step += 1
            input = input.to(config.device)
            # Train step
            loss_dict = vae_train_step(input)

            # Each step store train losses
            for k, v in loss_dict.items():
                train_losses[k].append(v.item())

            if config.step % config.log_frequency == 0:  # or i_iter == 10 or i_iter == 50 or i_iter == 100:
                # Print training loss
                log_msg = " - ".join([f'{k}: {np.mean(v):.4f}' for k,
                                      v in train_losses.items()])
                log_msg = f"Iteration {config.step} - " + log_msg
                log_msg += f" - time: {time() - t_start:.2f}s"
                print(log_msg)

                # Log to wandb
                log(
                    {f'train/{k}': np.mean(v) for k, v in train_losses.items()},
                    config
                )

                # Reset loss dict
                train_losses = defaultdict(list)

            if config.step % config.val_frequency == 0:  # or i_iter == 10 or i_iter == 50 or i_iter == 100:
                validate(val_loader, config)

            if config.step % config.anom_val_frequency == 0:
                # or i_iter == 10 or i_iter == 50 or i_iter == 100:
                evaluate(config, small_testloader, vae_val_step, val_loader)

            if config.step % config.save_frequency == 0:
                save_model(model, config, i_iter=config.step)

            if config.step >= config.max_steps:
                save_model(model, config)
                print(f'Reached {config.max_steps} iterations. Finished training {config.name}.')
                return

        i_epoch += 1
        print(f'Finished epoch {i_epoch}, ({config.step} iterations)')


if __name__ == '__main__':
    if config.eval:
        print('Evaluating model...')
        evaluate(config, big_testloader, vae_val_step, val_loader)

    else:
        train(model)
