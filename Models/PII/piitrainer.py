import sys

sys.path.append('/home/ioannis/Thesis/')
from argparse import ArgumentParser
import numpy as np
import torch
from Utilities.metrics import compute_average_precision
from Models.PII.PIImodel import WideResNetAE
import torch.nn.functional as F
from time import time
import wandb
from Utilities.common_config import common_config
from Utilities.evaluate import eval_dfr_pii
from Utilities.utils import (save_model,
                             seed_everything,
                             load_data,
                             # load_pretrained,
                             misc_settings,
                             load_model)

""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""


def get_config():
    parser = ArgumentParser()
    parser = common_config(parser)

    # Training Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--max_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    # Model Hyperparameters
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--k', type=int, default=4, help='k factor of Wide ResNet')

    return parser.parse_args()


config = get_config()

msg = "num_images_log should be lower or equal to batch size"
assert (config.batch_size >= config.num_images_log), msg

# Select training device
config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# get logger and naming string
config.method = 'PII'
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
model = WideResNetAE(config).to(config.device)

# Init optimizer, learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

# Load saved model  to evaluate
if config.eval:
    load_model(model, config)
    print('Saved model loaded.')

""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""


def train_step(model, optimizer, x, y):
    model.train()
    optimizer.zero_grad()
    anomaly_map = model(x)
    loss = F.binary_cross_entropy(anomaly_map, y)
    loss.backward()
    optimizer.step()
    return loss.item(), anomaly_map


def val_step(model, x, y):
    model.eval()
    with torch.no_grad():
        anomaly_map = model(x)
        loss = F.binary_cross_entropy(anomaly_map, y)
    return loss.item(), anomaly_map


def validate(model, val_loader, i_iter, config) -> None:
    val_losses = []
    i_val_step = 0

    for input, mask in val_loader:
        i_val_step += 1
        input = input.to(config.device)
        mask = mask.to(config.device)
        loss, anomaly_map = val_step(model, input, mask)
        val_losses.append(loss)
        # Print training loss
        if i_val_step >= config.val_steps:
            break

    log_msg = f"Iteration {i_iter} - "
    log_msg += f"validation loss: {np.mean(val_losses):.4f} - "

    print(log_msg)

    logger.log(
        {'val/loss': np.mean(val_losses),
         }, step=i_iter
    )

    return np.mean(val_losses)


def anom_val_step(model, input):
    model.eval()
    with torch.no_grad():
        anomaly_map = model(input).mean(1, keepdim=True)

    if config.modality == 'MRI':

        mask = torch.cat([inp > inp.min() for inp in input]).unsqueeze(1)
        anomaly_map *= mask

    anomaly_score = torch.tensor([anom.mean() for anom in anomaly_map])
    return anomaly_map, anomaly_score


def train(model):

    print('Starting training PII...')
    i_iter = 0
    i_epoch = 0

    if config.load_saved:
        i_iter = config.saved_iter

    train_losses = []
    anomaly_maps = []
    masks = []

    t_start = time()
    while True:

        for input, mask in train_loader:

            i_iter += 1
            input = input.to(config.device)
            mask = mask.to(config.device)
            # Train step
            loss, anomaly_map = train_step(model, optimizer, input, mask)

            # Add to losses
            train_losses.append(loss)
            anomaly_maps.append(anomaly_map.detach().cpu())
            masks.append(torch.where((mask).abs() > 0, 1, 0).cpu())
            if i_iter % config.log_frequency == 0:

                # Print training loss
                log_msg = f"Iteration {i_iter} - "
                log_msg += f"train loss: {np.mean(train_losses):.4f} - "

                # Due to it being expensive, calculate and log AP every second log step
                if i_iter % (4 * config.log_frequency) == 0:
                    pixel_ap = compute_average_precision(
                        torch.cat(anomaly_maps), torch.cat(masks))
                    log_msg += f"train pixel-ap: {pixel_ap:.4f} - "
                    logger.log(
                        {'train/pixel-ap': pixel_ap,
                         }, step=i_iter
                    )

                log_msg += f"time: {time() - t_start:.2f}s"
                print(log_msg)

                # Log to wandb

                logger.log(
                    {'train/loss': np.mean(train_losses),
                     }, step=i_iter
                )

                # log some images
                input_images = list(input[:config.num_images_log].cpu())
                anomaly_images = list(anomaly_map[:config.num_images_log].cpu())
                targets = list(mask[:config.num_images_log].cpu())

                logger.log({

                    'train/input images': [wandb.Image(img) for img in input_images],
                    'train/targets': [wandb.Image(img) for img in targets],
                    'train/anomaly maps': [wandb.Image(img) for img in anomaly_images],
                }, step=i_iter)

                # Reset loss dict
                train_losses = []
                anomaly_maps = []
                masks = []

            if i_iter % config.anom_val_frequency == 0:
                eval_dfr_pii(model, small_testloader, i_iter, anom_val_step, logger, config)

            if i_iter % config.val_frequency == 0:
                validate(model, val_loader, i_iter, config)

            if i_iter % config.save_frequency == 0 and i_iter != 0:
                save_model(model, config)

            if i_iter >= config.max_steps:
                save_model(model, config)
                print(f'Reached {config.max_steps} iterations. Finished training {config.naming_str}.')
                return

        i_epoch += 1
        print(f'Finished epoch {i_epoch}, ({i_iter} iterations)')


if __name__ == '__main__':
    if config.eval:
        print('Evaluating model...')
        eval_dfr_pii(model, big_testloader, 0, anom_val_step, logger, config)

    else:
        train(model)
