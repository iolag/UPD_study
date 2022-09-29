
import sys
import os
sys.path.append(os.path.expanduser('~/thesis/UAD_study/'))
from argparse import ArgumentParser
import numpy as np
import torch
import random
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from time import time
from torch import Tensor
from typing import Tuple
from unet import UNet
from scipy.ndimage import gaussian_filter
from Utilities.common_config import common_config
from Utilities.utils import (save_model, seed_everything,
                             load_data, load_pretrained,
                             misc_settings, ssim_map,
                             load_model, log, str_to_bool)
from Utilities.evaluate import evaluate
""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""


def get_config():

    parser = ArgumentParser()
    parser = common_config(parser)

    # Training Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay')
    parser.add_argument('--max_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr_schedule', type=str_to_bool, default=True, help='Use learning rate schedule.')

    # Model Hyperparameters
    parser.add_argument("--noise_res", type=float, default=16, help="noise resolution.")
    parser.add_argument("--noise_std", type=float, default=0.2, help="noise magnitude.")

    return parser.parse_args()


config = get_config()
config.method = 'DAE'
#config.center = True
misc_settings(config)

""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""

# specific seed for deterministic dataloader creation
seed_everything(42)

train_loader, val_loader, big_testloader, small_testloader = load_data(config)

""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""
# Reproducibility
seed_everything(config.seed)

print("Initializing model...")
model = UNet(in_channels=3, n_classes=config.img_channels).to(config.device)

# Load pretrained encoder
if config.load_pretrained and not config.eval:
    config.arch = 'unet'
    model = load_pretrained(model, config)

# Init optimizer, learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, amsgrad=True,
                             weight_decay=config.weight_decay)

lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=100)

# Load saved model to evaluate
if config.eval:
    model.load_state_dict(load_model(config))
    print('Saved model loaded.')

if config.speed_benchmark:
    from torchinfo import summary
    a = summary(model, (16, 3, 128, 128), verbose=0)
    params = a.total_params
    macs = a.total_mult_adds
    print('Number of Million parameters: ', params / 1e06)
    print('Number of GMACs: ', macs / 1e09)
""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""

# print(model.forward_down_flatten(next(iter(train_loader)).cuda())[0].shape)
# exit(0)


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
    if config.center:
        ns = (ns - 0.5) * 2
    # ns = ns.expand(-1, 3, -1, -1)W
    res = input + ns

    return res, ns


def train_step(input, noisy_input) -> Tuple[float, Tensor, Tensor]:
    model.train()
    optimizer.zero_grad()
    reconstruction = model(noisy_input)
    anomaly_map = torch.pow(reconstruction - input, 2).mean(1, keepdim=True)
    loss = anomaly_map.mean()
    loss.backward()
    optimizer.step()
    return loss.item()


def anom_val_step(input, return_loss: bool = True) -> Tuple[dict, Tensor]:

    model.eval()
    with torch.no_grad():
        # forward pass
        input_recon = model(input)
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

    # for MRI, RF apply brainmask
    if config.modality in ['MRI', 'CT']:
        mask = torch.stack([inp > inp.min() for inp in input])

        if config.get_images:
            anomaly_map *= mask
            mins = [(map[map > map.min()]) for map in anomaly_map]
            mins = [map.min() for map in mins]

            anomaly_map = torch.cat([(map - min) for map, min in zip(anomaly_map, mins)]).unsqueeze(1)

        anomaly_map *= mask
        anomaly_score = torch.tensor([map[inp > inp.min()].max() for map, inp in zip(anomaly_map, input)])

    elif config.modality in ['RF'] and config.dataset == 'DDR':
        anomaly_score = torch.tensor([map.max() for map in anomaly_map])
    else:
        anomaly_score = torch.tensor([map.mean() for map in anomaly_map])

    return anomaly_map, anomaly_score, input_recon


def train():
    print('Starting training DAE...')

    i_epoch = 0
    train_losses = []
    t_start = time()

    while True:
        for input in train_loader:
            config.step += 1
            input = input.to(config.device)
            noisy_input, noise_tensor = add_noise(input)
            loss = train_step(input, noisy_input)

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
                # Reset
                train_losses = []
            if config.step % 32 == 0 and config.lr_schedule:
                lr_scheduler.step()

            if config.step % config.anom_val_frequency == 0:
                # or i_iter == 10 or i_iter == 50 or i_iter == 100:
                evaluate(config, small_testloader, anom_val_step, val_loader)

            if config.step % config.save_frequency == 0:
                save_model(model, config, i_iter=config.step)

            if config.step >= config.max_steps:
                save_model(model, config)
                print(f'Reached {config.max_steps} iterations. Finished training {config.name}.')
                return

        i_epoch += 1


if __name__ == '__main__':
    if config.eval:
        print('Evaluating model...')
        evaluate(config, big_testloader, anom_val_step, val_loader)

    else:
        train()