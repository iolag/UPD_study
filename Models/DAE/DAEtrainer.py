import sys
sys.path.append('/data_ssd/users/lagi/thesis/UAD_study/')
from argparse import ArgumentParser
import numpy as np
import torch
import random
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from time import time
import wandb
from torch import Tensor
from typing import Tuple
from unet import UNet
from Utilities.common_config import common_config
from Utilities.evaluate import eval_reconstruction_based
from Utilities.utils import (
    seed_everything,
    save_model,
    load_model,
    load_data,
    load_pretrained,
    ssim_map,
    misc_settings
)
""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""


def get_config():

    parser = ArgumentParser()
    parser = common_config(parser)

    # Training Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay')
    parser.add_argument('--max_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')

    # Model Hyperparameters
    parser.add_argument("-nr", "--noise_res", type=float, default=32, help="noise resolution.")
    parser.add_argument("-ns", "--noise_std", type=float, default=0.1, help="noise magnitude.")

    return parser.parse_args()


config = get_config()

msg = "num_images_log should be lower or equal to batch size"
assert (config.batch_size >= config.num_images_log), msg

# Select training device
config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# get logger and naming string
config.method = 'DAE'
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

print("Initializing model...")
model = UNet(in_channels=config.img_channels, n_classes=config.img_channels).to(config.device)

# Load pretrained encoder
if config.load_pretrained and not config.eval:
    model.encoder = load_pretrained(model.encoder, config)

# Init optimizer, learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, amsgrad=True,
                             weight_decay=config.weight_decay)

lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=100)

# Load saved model to continue training or to evaluate
if config.eval:
    load_model(model, config)
    print('Saved model loaded.')

""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""


def add_noise(input):
    """
    Generate and apply randomly translated noise to batch x
    """

    # to apply it in for rgb maybe not take diff noise for each channel? (input.shape[1] should be 1)
    ns = torch.normal(mean=torch.zeros(input.shape[0], input.shape[1], config.noise_res, config.noise_res),
                      std=config.noise_std).to(config.device)

    ns = F.interpolate(ns, size=config.image_size, mode='bilinear', align_corners=True)

    # Roll to randomly translate the generated noise.
    roll_x = random.choice(range(config.image_size))
    roll_y = random.choice(range(config.image_size))
    ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])

    # Use foreground mask for MRI, to only apply noise in the foreground.
    if config.modality == 'MRI':
        mask = torch.cat([inp > inp.min() for inp in input]).unsqueeze(1)
        ns *= mask

    # ns = ns.expand(-1, 3, -1, -1)W
    res = input + ns

    return res, ns


def dae_train_step(input, noisy_input) -> Tuple[float, Tensor, Tensor]:
    model.train()
    optimizer.zero_grad()
    reconstruction = model(noisy_input)
    anomaly_map = torch.pow(reconstruction - input, 2).mean(1, keepdim=True)
    loss = anomaly_map.mean()
    loss.backward()
    optimizer.step()
    return loss.item(), reconstruction, anomaly_map


def anom_val_step(model, input) -> Tuple[Tensor, Tensor, Tensor]:

    model.eval()
    with torch.no_grad():
        # forward pass
        input_recon = model(input)

        if config.ssim_eval:
            anomaly_map = ssim_map(input_recon, input)
        else:
            anomaly_map = torch.abs(input_recon - input).mean(1, keepdim=True)

        if config.modality == 'MRI':
            mask = torch.cat([inp > inp.min() for inp in input]).unsqueeze(1)
            anomaly_map *= mask
            input_recon *= mask

        anomaly_score = torch.tensor([map[inp > inp.min()].mean() for map, inp in zip(anomaly_map, input)])

    return anomaly_map, anomaly_score, input_recon


def train() -> None:

    print('Starting training DAE...')
    i_iter = 0
    i_epoch = 0
    train_losses = []
    t_start = time()

    while True:
        for input in train_loader:
            i_iter += 1
            input = input.to(config.device)
            noisy_input, noise_tensor = add_noise(input)
            loss, reconstruction, normal_input_residual = dae_train_step(input, noisy_input)

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

                # Reset
                train_losses = []

                # Log images to wandb
                input_images = list(input[:config.num_images_log].cpu())
                reconstructions = list(reconstruction[:config.num_images_log].cpu())
                noisy_images = list(noisy_input[:config.num_images_log].cpu())
                noise = list(noise_tensor.float()[:config.num_images_log].cpu())
                residuals = list(normal_input_residual[:config.num_images_log].cpu())
                logger.log({
                    'train/input images': [wandb.Image(img) for img in input_images],
                    'train/reconstructions': [wandb.Image(img) for img in reconstructions],
                    'train/noisy_images': [wandb.Image(img) for img in noisy_images],
                    'train/noise': [wandb.Image(img) for img in noise],
                    'train/residuals': [wandb.Image(img) for img in residuals],
                }, step=i_iter)

            # if i_iter % 32 == 0:
            #     lr_scheduler.step()

            if i_iter % config.anom_val_frequency == 0:
                eval_reconstruction_based(model, small_testloader, i_iter, dae_train_step, logger, config)

            if i_iter % config.save_frequency == 0:
                save_model(model, config)

            if i_iter >= config.max_steps:
                print(f'Reached {config.max_steps} iterations. Finished training {config.naming_str}.')
                save_model(model, config)
                return

        i_epoch += 1

        print(f'Finished epoch {i_epoch}, ({i_iter} iterations)')


if __name__ == '__main__':
    if config.eval:
        print('Evaluating model...')
        eval_reconstruction_based(model, big_testloader, 0, dae_train_step, logger, config)
    else:
        train()
