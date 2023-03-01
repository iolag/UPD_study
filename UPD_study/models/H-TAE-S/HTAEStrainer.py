from argparse import ArgumentParser
import numpy as np
import torch
from HTAESmodel import HTAES
from time import time
from torch import Tensor
from typing import Tuple
import pathlib
from torchinfo import summary
from UPD_study.utilities.evaluate import evaluate
from UPD_study.utilities.common_config import common_config
from UPD_study.utilities.utils import (save_model, test_inference_speed, seed_everything,
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
    parser.add_argument('--max_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('-bs', '--batch_size', type=int, default=12, help='Batch size')

    # Model Hyperparameters
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--filters', type=int, default=96)
    parser.add_argument('--transformer_layers', type=int, default=8)
    parser.add_argument('--patch_size', '-ps', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0, help='transformer dropout')
    return parser.parse_args()


config = get_config()


# set initial script settings
config.method = 'HTAES'
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

model = HTAES(config).to(config.device)

# load CCD pretrained encoder and bottleneck
if config.load_pretrained and not config.eval:
    config.arch = 'htae'
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
    a = summary(model, (16, 1, 128, 128), verbose=0)
    params = a.total_params
    print('Number of Million parameters: ', params / 1e06)
    exit(0)

""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""


def train_step(input) -> dict:
    """
    Training step
    """
    model.train()
    optimizer.zero_grad()
    input_recon = model(input)
    loss = torch.mean(torch.abs(input - input_recon))  # MAE loss ###THEY USE MSE??
    loss.backward()
    optimizer.step()
    return loss


def val_step(input, test_samples: bool = False) -> Tuple[dict, Tensor]:
    """
    Validation step on validation or evaluation (test samples == True) set.
    """
    model.eval()

    with torch.no_grad():
        input_recon = model(input)

    loss = torch.mean(torch.abs(input - input_recon))  # MAE loss

    # Anomaly map
    if config.ssim_eval:

        anomaly_map = ssim_map(input_recon, input)

    else:
        anomaly_map = (input - input_recon).abs().mean(1, keepdim=True)

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
        return loss, anomaly_map, anomaly_score, input_recon


def validate(val_loader, config):
    """
    Validation logic on normal validation set.
    """
    val_losses = []
    i_val_step = 0

    for input in val_loader:
        # x [b, 1, h, w]
        input = input.to(config.device)
        # Compute loss
        loss, _, _, _ = val_step(input)
        val_losses.append(loss.item())
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

    return np.mean(val_losses)


def train() -> None:
    """
    Main training logic
    """
    print(f'Starting training {config.name}...')
    train_losses = []
    t_start = time()

    while True:
        for input in train_loader:

            config.step += 1
            input = input.to(config.device)
            loss = train_step(input)
            train_losses.append(loss.item())

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
    if config.speed_benchmark:
        test_inference_speed(val_step)
        exit(0)
    if config.eval:
        print(f'Evaluating {config.name}...')
        evaluate(config, big_testloader, val_step)
    else:
        train()
