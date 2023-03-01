'''
adapted from https://github.com/dbbbbm/f-AnoGAN-PyTorch
'''
from argparse import ArgumentParser
import numpy as np
import torch
from collections import defaultdict
from time import time
from torch import Tensor
from typing import Tuple
from GANmodel import fAnoGAN, calc_gradient_penalty
from torch.nn import functional as F
import pathlib
import os
from UPD_study.utilities.common_config import common_config
from UPD_study.utilities.utils import (seed_everything, test_inference_speed, str_to_bool, ssim_map,
                                       load_data, load_pretrained,
                                       misc_settings, log, load_model)
from torchinfo import summary
from UPD_study.utilities.evaluate import evaluate
from scipy.ndimage import gaussian_filter
""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""


def get_config():
    parser = ArgumentParser()
    parser = common_config(parser)

    # Training Hyperparameters
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--lr_d', type=float, default=2e-4, help='Discriminator learning rate')
    parser.add_argument('--lr_e', type=float, default=5e-5, help='Encoder learning rate')
    parser.add_argument('--gp_weight', type=float, default=10., help='Gradient penalty weight')
    parser.add_argument('--feat_weight', type=float, default=1.,
                        help='Feature reconstruction weight during encoder training')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--max_steps_gan', type=int, default=10000, help='Number of WGAN training steps.')
    parser.add_argument('--max_steps_encoder', type=int, default=10000,
                        help='Number of Encoder training steps.')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')

    # Model settings : for consistency with the original f-anogan method
    # dim and latent_dim are the same for the Encoder and Discriminator
    parser.add_argument('--dim', type=int, default=64, help='Model width')
    parser.add_argument('--latent_dim', type=int, default=128, help='Size of the latent space')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate, A.1 appendix of paper')
    parser.add_argument('--critic_iters', type=int, default=1,
                        help='Num of critic iterations per generator iteration.')
    # Save, Load, Train part settings
    parser.add_argument('--train_encoder', '-te', type=str_to_bool,
                        default=True, help='enable encoder training')
    parser.add_argument('--train_gan', '-tg', type=str_to_bool, default=True, help='enable gan training')

    parser.add_argument('--enc_val_frequency', '-evf', type=int,
                        default=1000, help='encoder validation frequency')
    parser.add_argument('--gan_val_frequency', '-gvf', type=int,
                        default=1000, help='wgan validation frequency')

    return parser.parse_args()


config = get_config()

if config.modality == 'MRI':
    config.max_steps_gan = 30000

# set initial script settings
config.method = 'f-anoGAN'
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
model = fAnoGAN(config).to(config.device)

# Init optimizers
optimizer_g = torch.optim.Adam(model.G.parameters(), lr=config.lr,
                               betas=(0., 0.9), weight_decay=config.weight_decay)
optimizer_d = torch.optim.Adam(model.D.parameters(), lr=config.lr_d,
                               betas=(0., 0.9), weight_decay=config.weight_decay)
optimizer_e = torch.optim.RMSprop(model.E.parameters(), lr=config.lr_e,
                                  weight_decay=config.weight_decay)

# Load saved model to evaluate
if config.eval:
    g, d, e = load_model(config)
    model.G.load_state_dict(g)
    model.D.load_state_dict(d)
    model.E.load_state_dict(e)
    print('Saved model loaded.')

# For when we train WGAN and Encoder in two different runs, load only D and G
if not config.train_gan and not config.eval:
    g, d = load_model(config)
    model.G.load_state_dict(g)
    model.D.load_state_dict(d)
    print('Saved WGAN loaded.')

# load CCD pretrained encoder
if config.load_pretrained and not config.eval:
    config.arch = 'fanogan'
    model.E = load_pretrained(model.E, config)

# Space Benchmark
if config.space_benchmark:
    a = summary(model.E, (16, 1, 128, 128), verbose=0)
    b = summary(model.D, (16, 1, 128, 128), verbose=0)
    c = summary(model.G, input_data=[torch.randn(16, config.latent_dim).cuda(), 16], verbose=0)
    params = a.total_params + b.total_params + c.total_params
    print('Number of Million parameters: ', params / 1e06)
    exit(0)

""""""""""""""""""""""""""""""""" GAN Training """""""""""""""""""""""""""""""""


def set_requires_grad(model, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad = requires_grad


def train_step_gan(x_real) -> Tuple[dict, Tensor]:
    """
    WGAN train step.
    """
    model.train()

    # enable G gradient calc
    set_requires_grad(model.G, True)

    # Generate fake images
    x_fake = model.G(batch_size=x_real.shape[0])

    """ 1. Train Discriminator, maximize log(D(x)) + log(1 - D(G(z))) """

    x_fake = model.G(batch_size=x_real.shape[0])
    set_requires_grad(model.D, True)
    set_requires_grad(model.G, False)
    optimizer_d.zero_grad()

    # Discriminator loss (Wasserstein loss)
    loss_real = -model.D(x_real)[0].mean()
    loss_fake = model.D(x_fake.detach())[0].mean()
    adv_loss_d = loss_real + loss_fake

    # Gradient penalty
    loss_gp = calc_gradient_penalty(model.D, x_real, x_fake)

    # Combine losses and backward
    loss_D = adv_loss_d + config.gp_weight * loss_gp
    loss_D.backward()
    optimizer_d.step()
    if config.step % config.critic_iters == 0:
        """ 2. Train Generator, maximize log(D(G(z))) """
        set_requires_grad(model.D, False)
        set_requires_grad(model.G, True)
        optimizer_g.zero_grad()

        # Generator loss
        pred_fake = model.D(x_fake)[0]
        adv_loss_g = -pred_fake.mean()

        loss_G = adv_loss_g
        loss_G.backward()
        optimizer_g.step()

    return {
        'd_loss_real': loss_real.item(),
        'd_loss_fake': loss_fake.item(),
        'adv_loss_d': adv_loss_d.item(),
        # 'adv_loss_g': adv_loss_g.item(),
        'loss_gp': loss_gp.item(),
        'loss_D': loss_D.item(),
    }, x_fake


def val_step_gan(x_real) -> Tuple[dict, Tensor]:
    """
    WGAN validation step on normal validation set.
    """
    model.eval()
    set_requires_grad(model.D, False)
    set_requires_grad(model.G, False)

    # Generate fake images

    x_fake = model.G(batch_size=x_real.shape[0])

    """ Only Critic loss required for validation """

    # Discriminator loss (Wasserstein loss)
    loss_real = -model.D(x_real)[0].mean()
    loss_fake = model.D(x_fake.detach())[0].mean()
    adv_loss_d = loss_real + loss_fake

    # Gradient penalty
    loss_gp = calc_gradient_penalty(model.D, x_real, x_fake)

    # Combine losses
    loss_D = adv_loss_d + config.gp_weight * loss_gp

    return {
        'd_loss_real': loss_real.item(),
        'd_loss_fake': loss_fake.item(),
        'adv_loss_d': adv_loss_d.item(),
        'loss_gp': loss_gp.item(),
        'loss_D': loss_D.item(),
    }, x_fake


def validate_gan() -> None:
    """
    WGAN  validation logic on normal validation set.
    """
    val_losses = defaultdict(list)
    i_val_step = 0

    for x_real in val_loader:
        x_real = x_real.to(config.device)

        loss_dict, _ = val_step_gan(x_real)

        for k, v in loss_dict.items():
            val_losses[k].append(v)

        i_val_step += 1

        if i_val_step >= config.val_steps:
            break

    # Print validation results
    log_msg = 'Validation: '
    log_msg += " - ".join([f'{k}: {np.mean(v):.4f}' for k, v in val_losses.items()])
    print(log_msg)

    # Log to wandb
    log({f'val/{k}': np.mean(v) for k, v in val_losses.items()}, config)


def train_gan():
    """
    WGAN training logic
    """
    print(f'Starting training {config.name}...')
    i_epoch = 0
    train_losses = defaultdict(list)
    t_start = time()

    while True:
        for x_real in train_loader:
            config.step += 1
            x_real = x_real.to(config.device)
            loss_dict, x_fake = train_step_gan(x_real)

            # Add to losses
            for k, v in loss_dict.items():
                train_losses[k].append(v)

            # log some real images for reference
            if config.step < 4:

                log({'train/real images': x_real}, config)

            if config.step % config.log_frequency == 0:
                # Print training loss
                log_msg = " - ".join([f'{k}: {np.mean(v):.4f}' for k,
                                      v in train_losses.items()])
                log_msg = f"Iteration {config.step} - " + log_msg
                log_msg += f" - time: {time() - t_start:.2f}s"

                print(log_msg)

                # Log to wandb
                log({f'train/{k}': np.mean(v)
                     for k, v in train_losses.items()}, config)

                log({'train/fake images': x_fake}, config)

                # Reset loss dict
                train_losses = defaultdict(list)

            # validate if normal_split != 1.0 so normal validation set is given
            if config.step % config.gan_val_frequency == 0:
                validate_gan()

            if config.step >= config.max_steps_gan:
                save_path = os.path.join(config.model_dir_path, 'saved_models')
                print(
                    f'Reached {config.max_steps_gan} iterations. Finished training GAN.')
                torch.save(model.D.state_dict(), f'{save_path}/{config.modality}/{config.name}_netD.pth')
                torch.save(model.G.state_dict(), f'{save_path}/{config.modality}/{config.name}_netG.pth')
                return

        i_epoch += 1
        print(f'Finished epoch {i_epoch}, ({config.step} iterations)')


""""""""""""""""""""""""""""""" Encoder Training """""""""""""""""""""""""""""""


def train_step_encoder(x):
    """
    Encoder training step
    """
    model.train()
    optimizer_e.zero_grad()

    z = model.E(x)  # encode image
    x_rec = model.G(z)  # decode latent vector
    x_feats = model.D.extract_feature(x)  # get features from real image
    # get features from reconstructed image
    x_rec_feats = model.D.extract_feature(x_rec)

    # Reconstruction loss
    loss_img = F.mse_loss(x_rec, x)
    loss_feats = F.mse_loss(x_rec_feats, x_feats)
    loss = loss_img + loss_feats * config.feat_weight

    loss.backward()
    optimizer_e.step()

    return {
        'loss_img_encoder': loss_img.item(),
        'loss_feats_encoder': loss_feats.item(),
        'loss_encoder': loss.item(),
    }, x_rec


def train_encoder():
    """
    Main training logic for the Encoder.
    """
    print(f'Starting training {config.name}...')
    config.step = 0
    train_losses = defaultdict(list)

    # Generator and discriminator don't require gradients
    set_requires_grad(model.D, False)
    set_requires_grad(model.G, False)

    t_start = time()

    while True:
        for x in train_loader:
            config.step += 1
            x = x.to(config.device)
            loss_dict, x_rec = train_step_encoder(x)

            # Add to losses
            for k, v in loss_dict.items():
                train_losses[k].append(v)

            if config.step % config.log_frequency == 0:
                # Print training loss
                log_msg = " - ".join([f'{k}: {np.mean(v):.4f}' for k,
                                      v in train_losses.items()])
                log_msg = f"Iteration {config.step} - " + log_msg
                log_msg += f" - time: {time() - t_start:.2f}s"
                print(log_msg)

                # Log to wandb
                log({f'train/{k}': np.mean(v)
                     for k, v in train_losses.items()},
                    config)

                # Reset loss dict
                train_losses = defaultdict(list)

            if config.step % config.enc_val_frequency == 0:
                _ = validate_encoder()

            if config.step % config.anom_val_frequency == 0:
                evaluate(config, small_testloader, val_step_encoder)

            if config.step >= config.max_steps_encoder:
                save_path = os.path.join(config.model_dir_path, 'saved_models')
                print(f'Reached {config.max_steps_encoder} iterations. Finished training encoder.')
                torch.save(model.E.state_dict(), f'{save_path}/{config.modality}/{config.name}_netE.pth')
                return


def val_step_encoder(input, test_samples: bool = False):
    model.eval()
    with torch.no_grad():

        # encode image
        z = model.E(input)

        # decode latent vector
        input_recon = model.G(z)

        # get features from real image
        x_feats = model.D.extract_feature(input)

        # get features from reconstructed image
        x_rec_feats = model.D.extract_feature(input_recon)

        # Reconstruction loss
        loss_img = F.mse_loss(input_recon, input)
        loss_feats = F.mse_loss(x_rec_feats, x_feats)
        loss = loss_img + loss_feats * config.feat_weight

        loss_dict = {
            'loss_img_encoder': loss_img.item(),
            'loss_feats_encoder': loss_feats.item(),
            'loss_encoder': loss.item(),
        }

        # Anomaly map
        if config.ssim_eval:
            anomaly_map = ssim_map(input, input_recon)
        else:
            anomaly_map = (input - input_recon).abs().mean(1, keepdim=True)

        if config.gaussian_blur:
            anomaly_map = anomaly_map.cpu().numpy()
            for i in range(anomaly_map.shape[0]):
                anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)
            anomaly_map = torch.from_numpy(anomaly_map).to(config.device)

        if config.modality == 'MRI':
            mask = torch.cat([inp > inp.min() for inp in input]).unsqueeze(1)
            anomaly_map *= mask

        # Anomaly score
        if config.ssim_eval:
            img_diff = ssim_map(input, input_recon)
        else:
            img_diff = (input - input_recon).pow(2)

        if config.gaussian_blur:
            img_diff = img_diff.cpu().numpy()
            for i in range(img_diff.shape[0]):
                img_diff[i] = gaussian_filter(img_diff[i], sigma=4)
            img_diff = torch.from_numpy(img_diff).to(config.device)

        if config.modality == 'MRI':
            mask = torch.cat([inp > inp.min() for inp in input]).unsqueeze(1)
            img_diff *= mask
            img_score = torch.tensor([map[inp > inp.min()].max() for map, inp in zip(img_diff, input)])

        elif config.modality == 'RF':
            img_score = torch.tensor([map.max() for map in anomaly_map])
        else:
            img_score = torch.tensor([map.mean() for map in anomaly_map])

        feat_diff = (x_feats - x_rec_feats).pow(2).mean((1))
        anomaly_score = img_score.to(config.device) + config.feat_weight * feat_diff

    if test_samples:
        return anomaly_map, anomaly_score, input_recon
    else:
        return loss_dict, anomaly_map, anomaly_score, input_recon


def validate_encoder() -> None:
    """
    Encoder validation logic on normal validation set.
    """

    val_losses = defaultdict(list)
    i_val_step = 0

    for input in val_loader:
        i_val_step += 1
        input = input.to(config.device)

        loss_dict, anomaly_map, _, input_recon = val_step_encoder(input)

        for k, v in loss_dict.items():
            val_losses[k].append(v)

        if i_val_step >= config.val_steps:
            break

    # Print validation results
    log_msg = 'Validation losses on normal samples: '
    log_msg += " - ".join([f'{k}: {np.mean(v):.4f}' for k, v in val_losses.items()])
    print(f'\n{log_msg}\n')

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

    return np.mean(val_losses['loss_encoder'])


if __name__ == '__main__':
    if config.speed_benchmark:
        test_inference_speed(val_step_encoder)
        exit(0)
    if config.eval:
        print('Evaluating model...')
        evaluate(config, big_testloader, val_step_encoder)
    else:
        # Train WGAN
        if config.train_gan:
            train_gan()

        # reinit logger if we train both gan and enc in a single run
        if config.train_gan and config.train_encoder:
            misc_settings(config)

        # Train encoder
        if config.train_encoder:
            train_encoder()
