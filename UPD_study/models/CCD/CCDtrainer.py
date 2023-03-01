"""
adapted from: https://github.com/tianyu0207/CCD/
"""
import argparse
import torch
import numpy as np
import torch.nn as nn
from models import ContrastiveModel
from losses import SimCLRLoss
from collections import defaultdict
from time import time
import pathlib

from UPD_study.utilities.common_config import common_config
from UPD_study.utilities.utils import (save_model, seed_everything,
                                       misc_settings, str_to_bool,
                                       log)


""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""


def get_config():
    parser = argparse.ArgumentParser()
    parser = common_config(parser)

    parser.add_argument('--cls-augmentation', '-aug', type=str, default='cutperm',
                        help='Augmentation for the classification task',
                        choices=['cutperm', 'rotation', 'cutout', 'noise'])  # cutperm works best d

    parser.add_argument('--lr-schedule', '-sch', type=str_to_bool, default=True,
                        help='Train only with the SimClr task')
    parser.add_argument('--only-simclr', type=str_to_bool, default=False,
                        help='Train only with the SimClr task')
    parser.add_argument('--backbone-arch', '-arch', type=str, default='wide_resnet50_2', help='Backbone',
                        choices=['resnet18', 'resnet50', 'wide_resnet50_2',
                                 'vae', 'fanogan', 'vgg19', 'unet', 'pii', 'amc', 'expvae'])
    parser.add_argument('--cls-head-number', default=2, type=int)
    parser.add_argument('--lr', '-lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--max-epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--max-steps', '-ms', type=int, default=20000, help='Number of training steps')

    # VAE Hyperparameters
    parser.add_argument('--latent_dim', type=int, default=512, help='Model width')
    parser.add_argument('--num_layers', type=int, default=6, help='Model width')
    parser.add_argument('--width', type=int, default=16, help='First conv layer num of filters')
    parser.add_argument('--conv1x1', type=int, default=16,
                        help='Channel downsampling with 1x1 convs before bottleneck')
    parser.add_argument('--kernel_size', type=int, default=3, help='convolutional kernel size')
    parser.add_argument('--padding', type=int, default=1,
                        help='padding for consistent downsampling, set 2 if kernel_size == 5')
    parser.add_argument('--kl_weight', type=float, default=0.001, help='KL loss term weight')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Input Dropout like https://doi.org/10.1145/1390156.1390294')

    # AMCons Hyperparameters
    parser.add_argument("--zdim", default=32, type=int)
    parser.add_argument("--dense", default=True, type=bool)
    parser.add_argument("--n_blocks", default=4, type=int)
    parser.add_argument("--input_shape", default=[1, 128, 128], type=list)

    config = parser.parse_args()
    return config


config = get_config()

# Specific modality params (Default above are for MRI t2)
if config.modality == 'CXR':
    config.kl_weight = 0.0001
    config.num_layers = 6
    config.latent_dim = 256
    config.width = 16
    config.conv1x1 = 64


if config.modality == 'MRI' and config.sequence == 't1' or config.modality == 'RF':
    config.kl_weight = 0.0001
    config.num_layers = 6
    config.latent_dim = 512
    config.width = 32
    config.conv1x1 = 64

# Specific backbone architecture params
if config.backbone_arch == 'fanogan':
    config.latent_dim = 128

if config.backbone_arch in ['vgg19', 'fanogan', 'amc']:
    config.lr = 0.001

if config.backbone_arch in ['unet']:
    config.lr = 0.0001
    if config.modality == 'MRI':
        config.max_steps = 10000

if config.backbone_arch in ['resnet18']:
    config.center = True

# set initial script settings
config.model_dir_path = pathlib.Path(__file__).parents[0]
config.method = 'CCD'
misc_settings(config)


""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""

print('Loading dataset...')

if config.modality == 'MRI':
    from UPD_study.models.CCD.datasets.MRI_CCD import get_train_dataloader
elif config.modality == 'CXR':
    from UPD_study.models.CCD.datasets.CXR_CCD import get_train_dataloader
elif config.modality == 'RF':
    from UPD_study.models.CCD.datasets.RF_CCD import get_train_dataloader

train_dataloader = get_train_dataloader(config)

""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""
# Reproducibility
seed_everything(config.seed)

# Init model
print("Initializing model...")
model = ContrastiveModel(config, features_dim=128).to(config.device)

if config.backbone_arch == 'vae':
    model.backbone['encoder'] = model.backbone['encoder'].to(config.device)
    model.backbone['bottleneck'] = model.backbone['bottleneck'].to(config.device)


# Loss, Optimizer
criterion = SimCLRLoss(temperature=0.2)
optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=0.0003, momentum=0.9)

# Initiate CELoss module
celoss = nn.CrossEntropyLoss()


def adjust_lr(lr, optimizer, epoch, max_epochs):
    """
    Learning rate schedule.
    """
    eta_min = lr * (0.1 ** 3)
    lr = eta_min + (lr - eta_min) * (1 + np.cos(np.pi * epoch / max_epochs)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train():
    """
    Main training logic
    """
    start_epoch = 0
    lr = config.lr
    train_losses = defaultdict(list)
    t_start = time()

    print('Starting pre-training with CCD...')
    for epoch in range(start_epoch, config.max_epochs):

        if config.lr_schedule:
            # Adjust lr (cosine annealing)
            lr = adjust_lr(lr, optimizer, epoch, config.max_epochs)

        for i, batch in enumerate(train_dataloader):

            config.step += 1
            loss_dict = train_step(batch)

            # Accumulate losses
            for k, v in loss_dict.items():
                train_losses[k].append(v.item())

            # Log losses
            if config.step % config.log_frequency == 0:
                # Print training loss
                log_msg = " - ".join([f'{k}: {np.mean(v):.4f}' for k,
                                      v in train_losses.items()])
                log_msg = f"Iteration {config.step} - Epoch {epoch} - " + log_msg
                log_msg += f" - time: {time() - t_start:.2f}s"
                log_msg += f" - learning rate: {lr:6f}"
                print(log_msg)

                # log losses
                log({f'val/{k}': np.mean(v) for k, v in train_losses.items()}, config)

                # Reset loss dict
                train_losses = defaultdict(list)

            if config.step % config.max_steps == 0:
                print(f'Reached {config.max_steps} iterations. Finished pre-training with CCD.')
                save_model(model.backbone, config)
                return


def train_step(batch):
    """
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    model.train()
    optimizer.zero_grad()

    # get current batch
    images = batch['image']
    images_augmented = batch['image_augmented']
    b, c, h, w = images.size()

    # combine α(x) and α'(x)
    input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
    input_ = input_.view(-1, c, h, w)

    input_ = input_.to(config.device)
    output, logits = model(input_)
    output = output.view(b, 2, -1)
    labels = batch['target']

    labels = labels.to(config.device)
    labels = labels.repeat(2)
    loss_cla = celoss(logits, labels)
    loss_con = criterion(output)

    if config.only_simclr:

        loss_con.backward()
        optimizer.step()
        return {'loss_con': loss_con}

    else:
        loss = loss_con + loss_cla
        loss.backward()
        optimizer.step()
        return {'loss_cla': loss_cla, 'loss_con': loss_con, 'loss': loss}


if __name__ == '__main__':
    train()
