from argparse import ArgumentParser
import numpy as np
import torch
from collections import defaultdict
from expVAEmodel import ConvVAE
from time import time
from torch import Tensor
from typing import Tuple
import pathlib
from torchinfo import summary
from gradcam import GradCAM
from UPD_study.utilities.evaluate import evaluate
from UPD_study.utilities.common_config import common_config
from UPD_study.utilities.utils import (save_model, test_inference_speed, seed_everything,
                                       load_data, load_pretrained,
                                       misc_settings,
                                       load_model, log)

""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""


def get_config():
    parser = ArgumentParser()
    parser = common_config(parser)

    # Training Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--max_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')

    # Model Hyperparameters
    parser.add_argument('--latent_size', type=int, default=32, help='Latent dimension.')
    parser.add_argument('--target_layer', type=str, default='encoder.2', help='Target layer for gradcam.')

    return parser.parse_args()


config = get_config()

# set initial script settings
config.method = 'expVAE'
config.model_dir_path = pathlib.Path(__file__).parents[0]
misc_settings(config)


""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""

# specific seed for creating the dataloader
seed_everything(42)

train_loader, val_loader, big_testloader, small_testloader = load_data(config)


""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""
# Reproducibility
seed_everything(config.seed)

print("Initializing model...")
model = ConvVAE(config.latent_size).to(config.device)

# initialize attention anomaly map generation tool
gcam = GradCAM(model, target_layer=config.target_layer, cuda=(config.device == 'cuda'))

# load CCD pretrained encoder and bottleneck
if config.load_pretrained and not config.eval:
    config.arch = 'expvae'
    model.encoder = load_pretrained(model.encoder, config)

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

    # Anomaly map
    input_recon, mu, logvar = gcam.forward(input)
    model.zero_grad()
    gcam.backward()
    anomaly_map = gcam.generate().detach()

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

            # if config.step % config.val_frequency == 0:
            #     validate(val_loader, config)

            if config.step % config.anom_val_frequency == 0:
                evaluate(config, small_testloader, vae_val_step)

            if config.step >= config.max_steps:
                save_model(model, config)
                print(f'Reached {config.max_steps} iterations. Finished training {config.name}.')
                return


if __name__ == '__main__':
    if config.speed_benchmark:
        test_inference_speed(vae_val_step)
        exit(0)
    if config.eval:
        print(f'Evaluating {config.name}...')
        evaluate(config, big_testloader, vae_val_step)
    else:
        train()
