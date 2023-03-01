from argparse import ArgumentParser
import numpy as np
import torch
from UPD_study.models.PII.PIImodel import WideResNetAE
import torch.nn.functional as F
from time import time
import pathlib
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
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    # Model Hyperparameters
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--k', type=int, default=4, help='k factor of Wide ResNet')

    return parser.parse_args()


config = get_config()

# set initial script settings
config.method = 'PII'
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
model = WideResNetAE(config).to(config.device)

# load CCD pretrainerd encoder
if config.load_pretrained and not config.eval:
    config.arch = 'pii'
    model = load_pretrained(model, config)

# Init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

# Load saved model to evaluate
if config.eval:
    model.load_state_dict(load_model(config))
    print('Saved model loaded.')

# Space Benchmark
if config.space_benchmark:
    from torchinfo import summary
    a = summary(model, (16, 3, 128, 128), verbose=0)
    params = a.total_params
    print('Number of Million parameters: ', params / 1e06)
    exit(0)

""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""


def train_step(x, y):
    """
    Training step
    """
    model.train()
    optimizer.zero_grad()
    anomaly_map = model(x)
    loss = F.binary_cross_entropy(anomaly_map, y)
    loss.backward()
    optimizer.step()
    return loss.item(), anomaly_map


def val_step(x, y):
    """
    Validation step
    """
    model.eval()
    with torch.no_grad():
        anomaly_map = model(x)
        loss = F.binary_cross_entropy(anomaly_map, y)
    return loss.item(), anomaly_map


def validate() -> None:
    """
    Validation logic on normal validation set.
    """
    val_losses = []
    i_val_step = 0

    for input, mask in val_loader:
        i_val_step += 1
        input = input.to(config.device)
        mask = mask.to(config.device)
        loss, anomaly_map = val_step(input, mask)
        val_losses.append(loss)
        # Print training loss
        if i_val_step >= config.val_steps:
            break

    log_msg = f"Iteration {config.step} - "
    log_msg += f"validation loss: {np.mean(val_losses):.4f} - "

    print(log_msg)

    log(
        {'val/loss': np.mean(val_losses),
         },
        config
    )

    # log images and residuals

    log({
        'val/input': input,
        'val/res': anomaly_map,
        'val/mask': mask,
    }, config)

    return np.mean(val_losses)


def anom_val_step(input, test_samples: bool = False):
    """
    Validation step for evaluation set.
    """
    model.eval()
    with torch.no_grad():
        anomaly_map = model(input).mean(1, keepdim=True)

    if config.modality == 'MRI':
        mask = torch.stack([inp > inp.min() for inp in input])
        anomaly_map *= mask
        anomaly_score = torch.tensor([map[inp > inp.min()].max() for map, inp in zip(anomaly_map, input)])

    elif config.modality == 'RF':
        anomaly_score = torch.tensor([map.max() for map in anomaly_map])
    else:
        anomaly_score = torch.tensor([map.mean() for map in anomaly_map])

    return anomaly_map, anomaly_score


def train():
    """
    Main training logic
    """
    print(f'Starting training {config.name}...')
    train_losses = []
    t_start = time()

    while True:

        for input, mask in train_loader:

            config.step += 1
            input = input.to(config.device)
            mask = mask.to(config.device)
            # Train step
            loss, anomaly_map = train_step(input, mask)

            # Add to losses
            train_losses.append(loss)

            if config.step % config.log_frequency == 0:

                # Print training loss
                log_msg = f"Iteration {config.step} - "
                log_msg += f"train loss: {np.mean(train_losses):.4f} - "

                # Due to it being expensive, calculate and log AP every second log step
                # if config.step % (4 * config.log_frequency) == 0:
                #     pixel_ap = compute_average_precision(
                #         torch.cat(anomaly_maps), torch.cat(masks))
                #     log_msg += f"train pixel-ap: {pixel_ap:.4f} - "
                #     log(
                #         {'train/pixel-ap': pixel_ap,
                #          }, config
                #     )

                log_msg += f"time: {time() - t_start:.2f}s"
                print(log_msg)

                # Log to wandb
                log(
                    {'train/loss': np.mean(train_losses),
                     }, config
                )
                # Reset loss dict
                train_losses = []

            if config.step % config.anom_val_frequency == 0:
                evaluate(config, small_testloader, anom_val_step)

            if config.step % config.val_frequency == 0:
                validate()

            if config.step >= config.max_steps:
                save_model(model, config)
                print(f'Reached {config.max_steps} iterations. Finished training {config.name}.')
                return


if __name__ == '__main__':
    if config.speed_benchmark:
        test_inference_speed(anom_val_step)
        exit(0)

    if config.eval:
        print('Evaluating model...')
        evaluate(config, big_testloader, anom_val_step)

    else:
        train()
