"""
Adapted from https://github.com/hq-deng/RD4AD
"""
import torch
import numpy as np
from torch.nn import functional as F
from argparse import ArgumentParser
from time import time
from typing import Tuple
from torch import Tensor
from scipy.ndimage import gaussian_filter
from torchinfo import summary
import pathlib
from resnet import resnet18, wide_resnet50_2
from de_resnet import de_resnet18, de_wide_resnet50_2
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
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--max_steps', '-ms', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--batch_size', '-bs', type=int, default=4, help='Batch size')

    # Model settings
    parser.add_argument('--arch', type=str, default='wide_resnet50_2',
                        choices=['resnet18', 'wide_resnet50_2'])

    return parser.parse_args()


config = get_config()

# set initial script settings
config.method = 'RD'
config.model_dir_path = pathlib.Path(__file__).parents[0]
misc_settings(config)

""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""

# specific seed for creating the dataloader
seed_everything(42)

train_loader, val_loader, big_testloader, small_testloader = load_data(config)

""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""

# Reproducibility
seed_everything(config.seed)

# Init model
print("Initializing model...")
if config.arch == 'resnet18':
    encoder, bn = resnet18(pretrained=True)
    decoder = de_resnet18(pretrained=False)
else:
    encoder, bn = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False)


encoder = encoder.to(config.device)
bn = bn.to(config.device)
encoder.eval()
decoder = decoder.to(config.device)

# load CCD pretrainerd encoder
if config.load_pretrained:
    encoder = load_pretrained(encoder, config)
    encoder.eval()

# dict that will be used for saving the model
model = {'encoder': encoder, 'decoder': decoder, 'bn': bn}

# Init optimizer
optimizer = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()),
                             lr=config.lr, betas=(0.5, 0.999))


# Load saved model to evaluate
if config.eval:
    load_dec, load_bn = load_model(config)
    decoder.load_state_dict(load_dec)
    bn.load_state_dict(load_bn)
    print('Saved model loaded.')

# Space Benchmark
if config.space_benchmark:
    input = next(iter(big_testloader))[0].cuda()
    a = summary(encoder, (16, 3, 128, 128), verbose=0)
    b = summary(bn, input_data=[encoder(input)], verbose=0)
    c = summary(decoder, bn(encoder(input)).shape, verbose=0)
    params = a.total_params + b.total_params + c.total_params
    print('Number of Million parameters: ', params / 1e06)
    exit(0)

""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""

cos_loss = torch.nn.CosineSimilarity()


def loss_fucntion(a, b):
    """
    Loss based on Cosine similarity between different encoder-decoder levels
    """
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                        b[item].view(b[item].shape[0], -1)))
    return loss


def train_step(input) -> float:
    """
    Training step
    """
    bn.train()
    decoder.train()
    optimizer.zero_grad()
    # Get encoder featmaps (list of featmaps of layer1, 2 and 3 of resnet)
    with torch.no_grad():
        enc_output = encoder(input)
    # Get decoder featmaps (list of featmaps of layer3, 2 and 1 of resnet)
    dec_output = decoder(bn(enc_output))
    loss = loss_fucntion(enc_output, dec_output)
    loss.backward()
    optimizer.step()
    return loss.item()


def get_anomaly_map(enc_output, dec_output, config) -> Tensor:
    """
    Generate anomaly map using cosine similarity between different encoder-decoder levels
    """
    anomaly_map = torch.zeros(enc_output[0].shape[0], 1, config.image_size,
                              config.image_size).to(config.device)

    for i in range(len(enc_output)):
        enc_feat_map = enc_output[i]
        dec_feat_map = dec_output[i]
        a_map = 1 - F.cosine_similarity(enc_feat_map, dec_feat_map)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=config.image_size, mode='bilinear', align_corners=True)
        anomaly_map += a_map

    # apply gaussian smoothing on the score map
    if config.gaussian_blur:
        anomaly_map = anomaly_map.detach().cpu().numpy()
        for i in range(anomaly_map.shape[0]):
            anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)

        anomaly_map = torch.from_numpy(anomaly_map).to(config.device)
    return anomaly_map


@ torch.no_grad()
def val_step(input, test_samples: bool = False) -> Tuple[float, Tensor, Tensor]:
    """
    Validation step on validation or evaluation (test samples == True) validation set.
    Calculates val loss, anomaly maps of shape batch_shape and anomaly scores of shape [b,1]
    """
    encoder.eval()
    decoder.eval()
    bn.eval()
    enc_output = encoder(input)  # [[b, 256, 32, 32], [b, 512, 16, 16], [b, 1024, 8, 8]]
    dec_output = decoder(bn(enc_output))  # [[b, 256, 32, 32], [b, 512, 16, 16], [b, 1024, 8, 8]]
    loss = loss_fucntion(enc_output, dec_output)
    anomaly_map = get_anomaly_map(enc_output, dec_output, config)

    # activations = [enc_output[0][-2, 0:10], enc_output[1][-2, 0:10], enc_output[2][-2, 0:10]]
    if config.modality == 'MRI':
        mask = torch.stack([inp[0].unsqueeze(0) > inp[0].min() for inp in input])
        anomaly_map *= mask
        anomaly_score = torch.tensor([map[inp[0].unsqueeze(0) > inp[0].min()].max()
                                      for map, inp in zip(anomaly_map, input)])

    elif config.modality == 'RF':
        anomaly_score = torch.tensor([map.max() for map in anomaly_map])
    else:
        anomaly_score = torch.tensor([map.mean() for map in anomaly_map])

    if test_samples:
        return anomaly_map, anomaly_score
    else:
        return loss.item(), anomaly_map, anomaly_score


def validate(val_loader, config):
    """
    Validation logic on normal validation set.
    """
    val_losses = []
    i_val_step = 0

    for input in val_loader:
        # x [b, 1, h, w]
        input = input.to(config.device)
        # Compute loss
        loss, anomaly_map, _ = val_step(input)
        val_losses.append(loss)
        i_val_step += 1

        if i_val_step >= config.val_steps:
            break

    # Print validation results
    log_msg = f"Validation loss on normal samples: {np.mean(val_losses):.4f}"
    print(log_msg)

    # Log to wandb
    config.logger.log({
        'val/loss': np.mean(val_losses),
    }, step=config.step)

    return np.mean(val_losses)


def train() -> None:
    """
    Main training logic
    """
    print(f'Starting training {config.name}...')
    train_losses = []
    t_start = time()

    while True:
        for input in train_loader:

            config.step += 1
            input = input.to(config.device)
            loss = train_step(input)
            train_losses.append(loss)

            if config.step % config.log_frequency == 0:

                # Print training loss
                log_msg = f"Iteration {config.step} - "
                log_msg += f"train loss: {np.mean(train_losses):.4f}"
                log_msg += f" - time: {time() - t_start:.2f}s"
                print(log_msg)

                # Log to wandb
                log({'train/loss': np.mean(train_losses)}, config)

                # Reset loss dict
                train_losses = []

            if config.step % config.val_frequency == 0:
                validate(val_loader, config)

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
