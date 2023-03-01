"""
adapted from https://github.com/FeliMe/feature-autoencoder
"""
from argparse import ArgumentParser
from collections import defaultdict
from time import time
import numpy as np
import torch
from UPD_study.models.FAE.FAEmodel import FeatureReconstructor
from torch import Tensor
from torchinfo import summary
from UPD_study.utilities.common_config import common_config
from UPD_study.utilities.utils import (save_model, test_inference_speed, seed_everything,
                                       load_data, load_pretrained,
                                       misc_settings, log, load_model)
from UPD_study.utilities.evaluate import evaluate
from typing import Tuple
import pathlib

""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""


def get_config():
    parser = ArgumentParser()
    parser = common_config(parser)

    # Training Hyperparameters

    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    # Model Hyperparameters
    parser.add_argument('--hidden_dims', type=int, nargs='+',
                        default=[100, 150, 200, 300],
                        help='Autoencoder hidden dimensions')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--loss_fn', type=str, default='ssim', help='loss function',
                        choices=['mse', 'ssim'])
    parser.add_argument('--extractor_cnn_layers', type=str, nargs='+',
                        default=['layer0', 'layer1', 'layer2'])
    parser.add_argument('--keep_feature_prop', type=float, default=1.0,
                        help='Proportion of ResNet features to keep')

    return parser.parse_args()


# set initial script settings
config = get_config()
config.method = 'FAE'
config.model_dir_path = pathlib.Path(__file__).parents[0]
config.center = True
misc_settings(config)


""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""

# specific seed for creating the dataloader
seed_everything(42)

train_loader, val_loader, big_testloader, small_testloader = load_data(config)

""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""
# Reproducibility
seed_everything(config.seed)

print("Initializing model...")
model = FeatureReconstructor(config).to(config.device)

# load CCD pretrained backbone
if config.load_pretrained:
    config.arch = 'resnet18'
    model.extractor.backbone = load_pretrained(model.extractor.backbone, config)

# Init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
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


def train_step(input) -> dict:
    """
    Training step
    """
    model.train()
    optimizer.zero_grad()
    loss_dict = model.loss(input)
    loss = loss_dict['rec_loss']
    loss.backward()
    optimizer.step()
    return loss_dict


def val_step(input, test_samples: bool = False) -> Tuple[dict, Tensor]:
    """
    Validation step on validation or evaluation (test samples == True) validation set.
    """
    model.eval()
    with torch.no_grad():
        loss_dict = model.loss(input)
        anomaly_map = model.predict_anomaly(input)

    # for MRI apply brainmask
    if config.modality == 'MRI':
        mask = torch.stack([inp > inp.min() for inp in input])
        anomaly_map *= mask
        anomaly_score = torch.tensor([map[inp > inp.min()].max() for map, inp in zip(anomaly_map, input)])

    elif config.modality == 'RF':
        anomaly_score = torch.tensor([map.max() for map in anomaly_map])

    elif config.modality == 'CXR':
        anomaly_score = torch.tensor([map.mean() for map in anomaly_map])

    if test_samples:
        return anomaly_map, anomaly_score
    else:
        return loss_dict, anomaly_map, anomaly_score


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

            if config.step % config.val_frequency == 0:
                validate(val_loader, config)

            if config.step % config.anom_val_frequency == 0:
                evaluate(config, small_testloader, val_step)

            if config.step >= config.max_steps:
                save_model(model, config)
                print(f'Reached {config.max_steps} iterations. Finished training {config.name}.')
                return


def validate(val_loader, config) -> None:
    """
    Validation logic on normal validation set.
    """
    val_losses = defaultdict(list)
    i_val_step = 0

    for input in val_loader:

        i_val_step += 1

        input = input.to(config.device)

        loss_dict, anomaly_map, anomaly_score = val_step(input)

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
        'val/res': anomaly_map,
    }, config)

    return np.mean(val_losses['rec_loss'])


if __name__ == '__main__':
    if config.speed_benchmark:
        test_inference_speed(val_step)
        exit(0)
    if config.eval:
        print(f'Evaluating {config.name}...')
        evaluate(config, big_testloader, val_step)
    else:
        train()
