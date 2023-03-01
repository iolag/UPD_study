"""
adapted from https://github.com/jusiro/constrained_anomaly_segmentation
"""
from argparse import ArgumentParser
from collections import defaultdict
from time import time
import numpy as np
from models import Encoder, Decoder
import torch
from torch import Tensor
from torchinfo import summary
import pathlib
from UPD_study.utilities.evaluate import evaluate
from UPD_study.utilities.common_config import common_config
from UPD_study.utilities.utils import (save_model, test_inference_speed, seed_everything,
                                       load_data, load_pretrained,
                                       misc_settings, load_model, log)

from typing import Tuple
import torch.nn.functional as F
""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""


def get_config():
    parser = ArgumentParser()
    parser = common_config(parser)

    # Hyper-params training
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='Number of training steps')

    # Auto-encoder architecture
    parser.add_argument("--zdim", default=32, type=int)
    parser.add_argument("--dense", default=True, type=bool)
    parser.add_argument("--n_blocks", default=4, type=int)

    # Settings with variational AE
    parser.add_argument("--wkl", default=10, type=float)

    # AMCons
    parser.add_argument("--wH", default=0.1, type=float, help='Alpha entropy')
    parser.add_argument("--level_cams", default=-4, type=float)

    return parser.parse_args()


config = get_config()

# set initial script settings
config.method = 'AMCons'
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

enc = Encoder(fin=3,
              zdim=config.zdim,
              dense=config.dense,
              n_blocks=config.n_blocks,
              spatial_dim=config.image_size // 2**config.n_blocks,
              variational=True,
              gap=False).to(config.device)

dec = Decoder(fin=config.zdim,
              nf0=enc.backbone.nfeats // 2,
              n_channels=config.img_channels,
              dense=config.dense,
              n_blocks=config.n_blocks,
              spatial_dim=config.image_size // 2**config.n_blocks,
              gap=False).to(config.device)

# Init optimizer
optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=config.lr)

# dict for saving the model
model = {'encoder': enc, 'decoder': dec}

# Load CCD pre-trained backbone
if config.load_pretrained and not config.eval:
    config.arch = 'amc'
    enc = load_pretrained(enc, config)
    enc.eval()

# Load saved model to evaluate
if config.eval:
    load_enc, load_dec = load_model(config)
    enc.load_state_dict(load_enc)
    dec.load_state_dict(load_dec)
    print('Saved model loaded.')

# Space Benchmark
if config.space_benchmark:
    input = next(iter(big_testloader))[0].cuda()
    a = summary(enc, (16, 3, 128, 128), verbose=0)
    b = summary(dec, enc(input)[0].shape, verbose=0)
    params = a.total_params + b.total_params
    print('Number of Million parameters: ', params / 1e06)
    exit(0)

""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""


def loss_kl(mu, logvar):
    """
    KL divergence loss
    """
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return kl_divergence


# Reconstruction Loss
loss_r = torch.nn.BCEWithLogitsLoss(reduction='sum')


def train_step(input) -> dict:
    """
    Training step
    """
    enc.train()
    dec.train()
    optimizer.zero_grad()

    # Forward pass
    z, z_mu, z_logvar, allF = enc(input)
    xhat, _ = dec(z)
    # Compute reconstruction and KL loss
    r_loss = loss_r(xhat, input) / config.batch_size
    kl_loss = loss_kl(mu=z_mu, logvar=z_logvar) / config.batch_size
    loss = r_loss + config.wkl * kl_loss

    # ---- Compute Attention Homogeneization loss via Entropy
    am = torch.mean(allF[config.level_cams], 1)
    # Restore original shape
    am = torch.nn.functional.interpolate(am.unsqueeze(1),
                                         size=(config.image_size, config.image_size),
                                         mode='bilinear',
                                         align_corners=True)

    # Probabilities
    p = torch.nn.functional.softmax(am.view((config.batch_size, -1)), dim=-1)
    # Mean entropy
    entropy_loss = torch.mean(-torch.sum(p * torch.log(p + 1e-12), dim=(-1)))
    loss += config.wH * entropy_loss
    loss.backward()
    optimizer.step()

    return {'loss': loss, 'rec_loss': r_loss, 'kl_loss': kl_loss, 'entropy_loss': entropy_loss}


def val_step(input, test_samples: bool = False) -> Tuple[dict, Tensor]:
    """
    Validation step on  evaluation set.
    """
    enc.eval()
    dec.eval()

    # Get reconstruction error map
    z, _, _, f = enc(input)
    input_recon = torch.sigmoid(dec(z)[0]).detach()

    anom_map_small = torch.mean(f[config.level_cams], 1)
    # Restore original shape
    anomaly_map = F.interpolate(anom_map_small.unsqueeze(1),
                                size=(config.image_size, config.image_size),
                                mode='bilinear',
                                align_corners=True).detach()

    # for MRI apply brainmask
    if config.modality == 'MRI':
        mask = torch.stack([inp > inp.min() for inp in input])
        anomaly_map *= mask
        anomaly_score = torch.tensor([map[inp > inp.min()].max() for map, inp in zip(anomaly_map, input)])

    elif config.modality == 'DDR':
        anomaly_score = torch.tensor([map.max() for map in anomaly_map])
    else:
        anomaly_score = torch.tensor([map.mean() for map in anomaly_map])

    return anomaly_map, anomaly_score, input_recon


def train():
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
            loss_dict = train_step(input)

            # Add to losses
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

                # Reset
                train_losses = defaultdict(list)

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
        print('Evaluating model...')
        evaluate(config, big_testloader, val_step)

    else:
        train()
