"""
adapted from: https://github.com/LilitYolyan/CutPaste
"""
from argparse import ArgumentParser
import numpy as np
import torch
from time import time
from torch import Tensor
from typing import Tuple
import pathlib
from UPD_study.models.CutPaste.CPmodel import CutPasteNet
from torchinfo import summary
from anomaly_detection import Detection
from localization import Localization
from UPD_study.utilities.evaluate import evaluate
from UPD_study.utilities.common_config import common_config
from UPD_study.utilities.utils import (test_inference_speed, save_model, seed_everything,
                                       load_data, load_pretrained,
                                       misc_settings, load_model,
                                       str_to_bool, log)

""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""


def get_config():
    parser = ArgumentParser()
    parser = common_config(parser)
    parser.add_argument('--fit_gde', default=False, type=str_to_bool,
                        help='Whether to fit GDE on normal data.')
    parser.add_argument('--align', default=True, type=str_to_bool, help='align')
    parser.add_argument('--dims', default=[512, 512, 512, 512, 512, 512, 512, 512, 128],
                        help='list indicating number of hidden units for each layer of projection head')
    parser.add_argument('--num_class', default=3)
    parser.add_argument('--encoder', default='resnet18')
    parser.add_argument('--cutpaste_type', default='3way')
    parser.add_argument('--lr', default=0.03)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--weight_decay', default=0.00003)
    parser.add_argument('--max_steps', default=10000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--localization', '-loc', default=True, type=str_to_bool,
                        help='If True train on (32,32) cropped patches and evaluate localization performance')
    return parser.parse_args()


config = get_config()

# set initial script settings
config.method = 'Cutpaste'
config.model_dir_path = pathlib.Path(__file__).parents[0]
misc_settings(config)


if config.localization:
    config.name += '_local'


""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""

# specific seed for creating the dataloader
seed_everything(42)

# use the actual, unaugmented train_loader to fit the GDE
# Covariance calculation cannot handle more than about 20% of CamCAN samples in our machine
if config.modality == 'MRI' and config.localization:
    temp = config.normal_split
    config.normal_split = 0.18

train_loader, val_loader, big_testloader, small_testloader = load_data(config)

if not config.eval:
    print('Loading Cutpaste dataset...')

    # reset original normal splits, as cutpaste samples are not used for covariance calculation
    if config.modality == 'MRI' and config.localization:
        config.normal_split = temp

    if config.modality == 'MRI':
        from UPD_study.models.CutPaste.datasets.MRI import cutpaste_loader
    elif config.modality == 'CXR':
        from UPD_study.models.CutPaste.datasets.CXR import cutpaste_loader
    elif config.modality == 'RF':
        from UPD_study.models.CutPaste.datasets.RF import cutpaste_loader

    # use the augmented train_loader for the actual training
    cutpaste_train_loader, cutpaste_val_loader = cutpaste_loader(config)

""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""
# Reproducibility
seed_everything(config.seed)

print("Initializing model...")
model = CutPasteNet(encoder=config.encoder,
                    pretrained=False,
                    dims=config.dims,
                    num_class=int(config.num_class)).to(config.device)

criterion = torch.nn.CrossEntropyLoss().to(config.device)


# load CCD pretrained encoder
if config.load_pretrained and not config.eval:
    config.arch = config.encoder
    model.encoder = load_pretrained(model.encoder, config)

# Init optimizer
optimizer = torch.optim.SGD(model.parameters(),
                            lr=config.lr,
                            momentum=config.momentum,
                            weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.max_steps)

# Load saved model to evaluate
if config.eval:
    model.load_state_dict(load_model(config))
    print('Saved model loaded.')

# init detection and localization classes
detection = Detection(model, train_loader, config)
localization = Localization(model, train_loader, config)

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
    x = torch.cat(input, axis=0).to(config.device)
    y = torch.arange(len(input))
    y = y.repeat_interleave(len(input[0])).to(config.device)
    logits, _ = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def val_step(input) -> dict:
    """
    Normal sample Validation step
    """
    model.eval()
    x = torch.cat(input, axis=0).to(config.device)
    y = torch.arange(len(input))
    y = y.repeat_interleave(len(input[0])).to(config.device)
    logits, _ = model(x)
    loss = criterion(logits, y)

    return loss


@torch.no_grad()
def eval_step(input, test_samples: bool = False) -> Tuple[dict, Tensor]:
    """
    Validation step on validation or evaluation (test samples == True) set.
    """
    if input.shape[1] == 1:
        input = input.repeat(1, 3, 1, 1)

    if config.localization:
        # Anomaly map

        anomaly_map = localization.anomaly_map(input)
        # for MRI apply brainmask
        if config.modality == 'MRI':
            mask = torch.stack([inp[0].unsqueeze(0) > inp[0].min() for inp in input])
            if config.get_images:
                anomaly_map *= mask
                mins = [(map[map > map.min()]) for map in anomaly_map]
                mins = [map.min() for map in mins]
                anomaly_map = torch.cat([(map - min) for map, min in zip(anomaly_map, mins)]).unsqueeze(1)
            anomaly_map *= mask
        anomaly_score = None
    else:
        anomaly_map = None
        anomaly_score = detection.GDE_scores(input)  # torch.from_numpy(detection.GDE_scores(input))

    return anomaly_map, anomaly_score


def validate() -> None:
    """
    Validation logic on normal validation set.
    """
    val_losses = []
    i_val_step = 0

    for input in cutpaste_val_loader:

        i_val_step += 1
        loss = val_step(input)
        val_losses.append(loss.item())

        if i_val_step >= config.val_steps:
            break

    # Print validation results
    validation_loss = np.mean(val_losses)
    log_msg = f'Validation loss on normal samples: {validation_loss}'
    print(log_msg)

    # Log to wandb
    log(
        {'val/val_loss': validation_loss},
        config
    )
    log({
        'val/original': input[0],
        'val/cutpaste': input[1],
        'val/scar': input[2],
    }, config)
    return validation_loss


def train() -> None:
    """
    Main training logic
    """
    print(f'Starting training {config.name}...')
    train_losses = []
    t_start = time()

    while True:

        for input in cutpaste_train_loader:

            config.step += 1

            # Train step
            loss = train_step(input)
            train_losses.append(loss.item())

            if config.step % config.log_frequency == 0:
                # Print training loss
                train_loss = np.mean(train_losses)
                log_msg = f'Train loss: {train_loss} - '
                log_msg += f'Iteration {config.step} - '
                log_msg += f' - time: {time() - t_start:.2f}s'
                print(log_msg)

                # Log to wandb
                log(
                    {'train/train_loss': train_loss},
                    config
                )
                # log images

                log({
                    'train/original': input[0],
                    'train/cutpaste': input[1],
                    'train/scar': input[2],
                }, config)

                # Reset loss dict
                train_losses = []

            if config.step % config.val_frequency == 0:
                validate()

            # if config.step % config.anom_val_frequency == 0:
            #     evaluate(config, small_testloader, eval_step)

            if config.step >= config.max_steps:

                save_model(model, config)
                print(f'Reached {config.max_steps} iterations. Finished training {config.name}.')
                print('Constructing {} Gaussian Density Estimator...'.format(
                    'local' if config.localization else 'global'))
                if config.localization:
                    localization.patch_GDE_fit()
                else:
                    detection.GDE_fit()
                return


if __name__ == '__main__':
    if config.speed_benchmark:
        if config.localization:
            localization.patch_GDE_fit()
        else:
            detection.GDE_fit()
        test_inference_speed(eval_step)
        exit(0)
    if config.eval:
        print(f'Evaluating {config.name}...')

        print('Constructing {} Gaussian Density Estimator...'.format(
            'local' if config.localization else 'global'))
        if config.localization:
            localization.patch_GDE_fit()
        else:
            detection.GDE_fit()
        evaluate(config, big_testloader, eval_step)
    else:
        train()
