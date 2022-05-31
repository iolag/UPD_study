import sys
sys.path.append('/home/ioannis/lagi/thesis/UAD_study')
from argparse import ArgumentParser
import numpy as np
import torch
from collections import defaultdict
from time import time
import wandb
from torch import Tensor
from typing import Tuple
from GANmodel import fAnoGAN, calc_gradient_penalty
from torch.nn import functional as F
from Utilities.evaluate import eval_reconstruction_based
from Utilities.common_config import common_config
from Utilities.utils import (set_requires_grad,
                             seed_everything,
                             load_data,
                             misc_settings,
                             str_to_bool,
                             ssim_map)

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
    parser.add_argument('--max_steps_gan', type=int, default=50000, help='Number of training steps')
    parser.add_argument('--max_steps_encoder', type=int, default=20000, help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    # Model settings : for consistency with the original f-anogan method
    # dim and latent_dim should be the same for the Encoder and GAN discriminator
    parser.add_argument('--dim', type=int, default=64, help='Model width')
    parser.add_argument('--latent_dim', type=int, default=128, help='Size of the latent space')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate, A.1 appendix of paper')
    parser.add_argument('--critic_iters', type=int, default=1, help='Num of critic iters per generator iter')
    # Save, Load, Train part settings
    parser.add_argument('--train_encoder', type=str_to_bool, default=False, help='enable encoder training')
    parser.add_argument('--train_gan', type=str_to_bool, default=False, help='enable gan training')
    parser.add_argument('--gan_iter', type=str, default="", help='Gan num of iters')

    return parser.parse_args()


config = get_config()

msg = "num_images_log should be lower or equal to batch size"
assert (config.batch_size >= config.num_images_log), msg

# Select training device
config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# get logger and naming string
config.method = 'f-anoGAN'
naming_str, logger = misc_settings(config)

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
model = fAnoGAN(config).to(config.device)

# Init optimizers
optimizer_g = torch.optim.Adam(model.G.parameters(), lr=config.lr,
                               betas=(0., 0.9), weight_decay=config.weight_decay)
optimizer_d = torch.optim.Adam(model.D.parameters(), lr=config.lr_d,
                               betas=(0., 0.9), weight_decay=config.weight_decay)
optimizer_e = torch.optim.RMSprop(model.E.parameters(), lr=config.lr_e, weight_decay=config.weight_decay)

if config.eval:
    model.G.load_state_dict(torch.load(f'saved_models/{config.modality}/{naming_str}_netG.pth'))
    model.D.load_state_dict(torch.load(f'saved_models/{config.modality}/{naming_str}_netD.pth'))
    model.E.load_state_dict(torch.load(f'saved_models/{config.modality}/{naming_str}_netE.pth'))
    print('Saved model loaded.')

""""""""""""""""""""""""""""""""" GAN Training """""""""""""""""""""""""""""""""


def train_step_gan(model, optimizer_g, optimizer_d, x_real, iter) -> Tuple[dict, Tensor]:

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
    if iter % config.critic_iters == 0:
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


def val_step_gan(model, x_real) -> Tuple[dict, Tensor]:
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


def validate_gan(model, i_iter, config) -> None:

    val_losses = defaultdict(list)
    i_val_step = 0

    for x_real in val_loader:
        x_real = x_real.to(config.device)

        loss_dict, _ = val_step_gan(model, x_real)

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
    logger.log(
        {f'val/{k}': np.mean(v)
         for k, v in val_losses.items()},
        step=i_iter
    )


def train_gan(model, config):

    print('Starting training GAN...')
    i_iter = 0
    i_epoch = 0
    train_losses = defaultdict(list)

    t_start = time()

    while True:
        for x_real in train_loader:
            i_iter += 1
            x_real = x_real.to(config.device)
            loss_dict, x_fake = train_step_gan(model, optimizer_g, optimizer_d, x_real, i_iter)

            # Add to losses
            for k, v in loss_dict.items():
                train_losses[k].append(v)

            # log some real images
            if i_iter < 4:
                real_images = list(x_real[:config.num_images_log].cpu().detach())
                real_images = [wandb.Image(image) for image in real_images]

                logger.log({
                    'train/real images': real_images
                }, step=i_iter)

            if i_iter % config.log_frequency == 0:
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

                fake_images = list(x_fake[:config.num_images_log].cpu().detach())
                fake_images = [wandb.Image(image) for image in fake_images]

                logger.log({
                    'train/fake images': fake_images
                }, step=i_iter)

                # Reset loss dict
                train_losses = defaultdict(list)

            # validate if normal_split != 1.0 so normal validation set is given
            if i_iter % config.gan_val_frequency == 0 and config.normal_split != 1.0:
                validate_gan(model, i_iter, config)

            if i_iter % config.save_frequency == 0 and i_iter != 0:
                torch.save(model.D.state_dict(),
                           f'saved_models/{config.modality}/{naming_str}_netD_{i_iter}.pth')
                torch.save(model.G.state_dict(),
                           f'saved_models/{config.modality}/{naming_str}_netG_{i_iter}.pth')

            if i_iter >= config.max_steps_gan:
                print(
                    f'Reached {config.max_steps_gan} iterations. Finished training GAN.')
                torch.save(model.D.state_dict(), f'saved_models/{config.modality}/{naming_str}_netD.pth')
                torch.save(model.G.state_dict(), f'saved_models/{config.modality}/{naming_str}_netG.pth')
                return

        i_epoch += 1
        print(f'Finished epoch {i_epoch}, ({i_iter} iterations)')


""""""""""""""""""""""""""""""" Encoder Training """""""""""""""""""""""""""""""


def train_step_encoder(model, optimizer_e, x, config):
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


def train_encoder(model, config):
    print('Starting training Encoder...')
    i_iter = 0
    i_epoch = 0
    if config.load_saved:
        i_iter = config.saved_iter

    train_losses = defaultdict(list)

    # Generator and discriminator don't require gradients
    set_requires_grad(model.D, False)
    set_requires_grad(model.G, False)

    t_start = time()

    while True:
        for x in train_loader:
            i_iter += 1
            x = x.to(config.device)
            loss_dict, x_rec = train_step_encoder(model, optimizer_e, x, config)

            # Add to losses
            for k, v in loss_dict.items():
                train_losses[k].append(v)

            if i_iter % config.log_frequency == 0:
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

                # Reset
                train_losses = defaultdict(list)

            if i_iter % config.enc_val_frequency == 0:
                _ = validate_encoder(model, val_loader, i_iter, config)

            if i_iter % config.anom_val_frequency == 0:
                eval_reconstruction_based(model, small_testloader, i_iter, val_step_encoder, logger, config)

            if i_iter % config.save_frequency == 0 and i_iter != 0:
                torch.save(model.E.state_dict(),
                           f'saved_models/{config.modality}/{naming_str}_netE_{i_iter}.pth')

            if i_iter >= config.max_steps_encoder:
                print(
                    f'Reached {config.max_steps_encoder} iterations. Finished training encoder.')
                torch.save(model.E.state_dict(),
                           f'saved_models/{config.modality}/{naming_str}feat_netE.pth')

                return

        i_epoch += 1
        print(f'Finished epoch {i_epoch}, ({i_iter} iterations)')


def val_step_encoder(model, input, config, return_loss: bool = True):
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
            ssim_map(input, input_recon)
        else:
            anomaly_map = (input - input_recon).abs().mean(1, keepdim=True)

        if config.modality == 'MRI':
            mask = torch.cat([inp > inp.min() for inp in input]).unsqueeze(1)
            anomaly_map *= mask

        # Anomaly score
        img_diff = input - input_recon

        if config.modality == 'MRI':
            img_diff *= mask

        img_diff = img_diff.pow(2).mean((1, 2, 3))
        feat_diff = (x_feats - x_rec_feats).pow(2).mean((1))
        anomaly_score = img_diff + config.feat_weight * feat_diff

    if return_loss:
        return loss_dict, anomaly_map, anomaly_score, input_recon
    else:
        return anomaly_map, anomaly_score, input_recon


def validate_encoder(model, val_loader, i_iter, config) -> None:

    val_losses = defaultdict(list)
    i_val_step = 0

    for input in val_loader:
        i_val_step += 1
        input = input.to(config.device)

        loss_dict, anomaly_map, _, input_recon = val_step_encoder(
            model, input, config)

        for k, v in loss_dict.items():
            val_losses[k].append(v)

        if i_val_step >= config.val_steps:
            break

    # Print validation results
    log_msg = 'Validation losses on normal samples: '
    log_msg += " - ".join([f'{k}: {np.mean(v):.4f}' for k, v in val_losses.items()])
    print(f'\n{log_msg}\n')

    # Log to wandb
    logger.log(
        {f'val/{k}': np.mean(v)
         for k, v in val_losses.items()},
        step=i_iter
    )

    # log images and residuals
    input_images = list(input[:config.num_images_log].cpu())
    input_images = [wandb.Image(image) for image in input_images]

    reconstructions = list(input_recon[:config.num_images_log].cpu())
    reconstructions = [wandb.Image(image) for image in reconstructions]

    residuals = list(anomaly_map[:config.num_images_log].cpu())
    residuals = [wandb.Image(image) for image in residuals]

    logger.log({
        'val/input': input_images,
        'val/recon': reconstructions,
        'val/res': residuals,
    }, step=i_iter)

    return np.mean(val_losses['loss_encoder'])


if __name__ == '__main__':
    if config.eval:
        config.num_images_log = 100
        print('Evaluating model...')
        eval_reconstruction_based(model, big_testloader, 0, val_step_encoder, logger, config)

    if config.train_gan:
        train_gan(model, config)

    if config.train_encoder:
        if not config.train_gan:
            model.G.load_state_dict(torch.load(f'saved_models/{config.modality}/{naming_str}_netG.pth'))
            model.D.load_state_dict(torch.load(f'saved_models/{config.modality}/{naming_str}_netD.pth'))
            print('Saved GAN model loaded.')

        train_encoder(model, config)
