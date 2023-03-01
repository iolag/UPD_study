"""
adapted from: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
"""
from argparse import ArgumentParser
import numpy as np
import torch
from collections import defaultdict
from VAEmodel import VAE
from time import time
from torch import Tensor
from typing import Tuple
import pathlib
from torchinfo import summary
from scipy.ndimage import gaussian_filter
from UPD_study.utilities.evaluate import evaluate
from UPD_study.utilities.common_config import common_config
from UPD_study.utilities.utils import (save_model, seed_everything,
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
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')

    # Model Hyperparameters
    parser.add_argument('--kl_weight', type=float, default=0.001, help='kl loss term weight')
    parser.add_argument('--latent_dim', type=int, default=512, help='Latent dimension.')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Numer of encoder (and decoder) conv layers')
    parser.add_argument('--width', type=int, default=16, help='First conv layer number of filters.')
    parser.add_argument('--conv1x1', type=int, default=16,
                        help='Channel downsampling with 1x1 convs before bottleneck.')
    parser.add_argument('--kernel_size', type=int, default=3, help='Convolutional kernel size.')
    parser.add_argument('--padding', type=int, default=1,
                        help='Padding for consistent downsampling, set to 2 if kernel_size == 5.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Input Dropout like https://doi.org/10.1145/1390156.1390294')

    return parser.parse_args()


config = get_config()

# set initial script settings
config.restoration = False
config.method = 'VAE'
config.model_dir_path = pathlib.Path(__file__).parents[0]
misc_settings(config)

# Specific modality params (Default are for MRI t2)
if config.modality != 'MRI':
    config.max_steps = 10000

if config.modality == 'CXR':
    config.kl_weight = 0.0001
    config.num_layers = 6
    config.latent_dim = 256
    config.width = 16
    config.conv1x1 = 64

if (config.modality == 'MRI' and config.sequence == 't1') or config.modality == 'RF':
    config.kl_weight = 0.0001
    config.num_layers = 6
    config.latent_dim = 512
    config.width = 32
    config.conv1x1 = 64

""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""

# specific seed for creating the dataloader
seed_everything(42)

train_loader, val_loader, big_testloader, small_testloader = load_data(config)


""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""
# Reproducibility
seed_everything(config.seed)

print("Initializing model...")
model = VAE(config).to(config.device)

# load CCD pretrained encoder and bottleneck
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

# Space Benchmark
if config.space_benchmark:
    a = summary(model, (16, 3, 128, 128), verbose=0)
    params = a.total_params
    print('Number of Million parameters: ', params / 1e06)
    exit(0)

""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""


def vae_train_step(input) -> dict:
    """
    Training step
    """
    model.train()
    optimizer.zero_grad()
    input_recon, mu, logvar = model(input)
    loss_dict = model.loss_function(input, input_recon, mu, logvar)  # VAE loss
    loss = loss_dict['loss']
    loss.backward()
    optimizer.step()
    return loss_dict


def vae_val_step(input, test_samples: bool = False) -> Tuple[dict, Tensor]:
    """
    Validation step on validation or evaluation (test samples == True) set.
    """
    model.eval()

    with torch.no_grad():
        input_recon, mu, logvar = model(input)

    loss_dict = model.loss_function(input, input_recon, mu, logvar)  # VAE Loss

    # Anomaly map
    if config.ssim_eval:

        anomaly_map = ssim_map(input_recon, input)

        if config.gaussian_blur:
            anomaly_map = anomaly_map.cpu().numpy()
            for i in range(anomaly_map.shape[0]):
                anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=config.sigma)
            anomaly_map = torch.from_numpy(anomaly_map).to(config.device)
    else:
        anomaly_map = (input - input_recon).abs().mean(1, keepdim=True)
        if config.gaussian_blur:
            anomaly_map = anomaly_map.cpu().numpy()
            for i in range(anomaly_map.shape[0]):
                anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=config.sigma)
            anomaly_map = torch.from_numpy(anomaly_map).to(config.device)

    # for MRI apply brainmask
    if config.modality == 'MRI':
        mask = torch.stack([inp[0].unsqueeze(0) > inp[0].min() for inp in input])
        anomaly_map *= mask
        anomaly_score = torch.tensor([map[inp[0].unsqueeze(0) > inp[0].min()].max()
                                     for map, inp in zip(anomaly_map, input)])

    elif config.modality == 'RF':
        anomaly_score = torch.tensor([map.max() for map in anomaly_map])
    else:
        anomaly_score = torch.tensor([map.mean() for map in anomaly_map])

    if test_samples:
        return anomaly_map, anomaly_score, input_recon
    else:
        return loss_dict, anomaly_map, anomaly_score, input_recon


def validate(val_loader, config) -> None:
    """
    Validation logic on normal validation set.
    """
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


def train() -> None:
    """
    Main training logic
    """
    print(f'Starting training {config.name}...')
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

            if config.step % config.log_frequency == 0:
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

            if config.step % config.val_frequency == 0:
                validate(val_loader, config)

            if config.step % config.anom_val_frequency == 0:
                evaluate(config, small_testloader, vae_val_step)

            if config.step >= config.max_steps:
                save_model(model, config)
                print(f'Reached {config.max_steps} iterations. Finished training {config.name}.')
                return


from UPD_study.utilities.utils import test_inference_speed
if __name__ == '__main__':
    if config.speed_benchmark:
        test_inference_speed(vae_val_step)
        exit(0)
    if config.eval:
        print(f'Evaluating {config.name}...')
        evaluate(config, big_testloader, vae_val_step)
    else:
        train()
