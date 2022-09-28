import sys
import os
sys.path.append(os.path.expanduser('~/thesis/UAD_study/'))
import torch
import numpy as np
from torch.nn import functional as F
from argparse import ArgumentParser
from time import time
from typing import Tuple
from torch import Tensor
from scipy.ndimage import gaussian_filter
from Utilities.common_config import common_config
from Utilities.evaluate import evaluate, evaluate_hist
from Utilities.utils import (save_model, seed_everything,
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
    # epochs = 200
    parser.add_argument('--max_steps', '-ms', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--batch_size', '-bs', type=int, default=4, help='Batch size')

    # Model settings
    parser.add_argument('--arch', type=str, default='wide_resnet50_2',
                        choices=['resnet18', 'wide_resnet50_2'])

    return parser.parse_args()


config = get_config()

config.method = 'RD'

# general setup
misc_settings(config)

""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""
# specific seed for same dataloader creation accross different seeds
seed_everything(42)

train_loader, val_loader, big_testloader, small_testloader = load_data(config)


""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""

# Reproducibility
seed_everything(config.seed)

# Init model
print("Initializing model...")
if config.arch == 'resnet18':
    if not config.deep:
        from resnet import resnet18  # ,resnet18, resnet50,
        from de_resnet import de_resnet18  # , de_resnet50,de_resnet18
    else:
        from deep.resnet import resnet18  # ,resnet18, resnet50,
        from deep.de_resnet import de_resnet18  # , de_resnet50,de_resnet18
    encoder, bn = resnet18(pretrained=True)
    decoder = de_resnet18(pretrained=False)
else:
    if not config.deep:
        from resnet import wide_resnet50_2  # ,resnet18, resnet50,
        from de_resnet import de_wide_resnet50_2  # , de_resnet50,de_resnet18
    else:
        from deep.resnet import wide_resnet50_2  # ,resnet18, resnet50,
        from deep.de_resnet import de_wide_resnet50_2  # , de_resnet50,de_resnet18
    encoder, bn = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False)


encoder = encoder.to(config.device)
bn = bn.to(config.device)
encoder.eval()
decoder = decoder.to(config.device)
# load pretrained with CCD
if config.load_pretrained:
    encoder = load_pretrained(encoder, config)
    encoder.eval()

# use this dict for saving
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

if config.speed_benchmark:
    from torchinfo import summary
    input = next(iter(big_testloader))[0].cuda()

    a = summary(encoder, (16, 3, 128, 128), verbose=0)
    b = summary(bn, input_data=[encoder(input)], verbose=0)
    c = summary(decoder, bn(encoder(input)).shape, verbose=0)
    params = a.total_params + b.total_params + c.total_params
    macs = a.total_mult_adds + b.total_mult_adds + c.total_mult_adds
    print('Number of Million parameters: ', params / 1e06)
    print('Number of GMACs: ', macs / 1e09)

""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""

cos_loss = torch.nn.CosineSimilarity()


def loss_fucntion(a, b):

    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                        b[item].view(b[item].shape[0], -1)))
    return loss


def train_step(input) -> float:
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
def val_step(input, return_loss: bool = True) -> Tuple[float, Tensor, Tensor]:
    """Calculates val loss, anomaly maps of shape batch_shape and anomaly scores of shape [b,1]"""
    encoder.eval()
    decoder.eval()
    bn.eval()
    enc_output = encoder(input)  # [[b, 256, 32, 32], [b, 512, 16, 16], [b, 1024, 8, 8]]
    dec_output = decoder(bn(enc_output))  # [[b, 256, 32, 32], [b, 512, 16, 16], [b, 1024, 8, 8]]
    loss = loss_fucntion(enc_output, dec_output)
    anomaly_map = get_anomaly_map(enc_output, dec_output, config)

    # activations = [enc_output[0][-2, 0:10], enc_output[1][-2, 0:10], enc_output[2][-2, 0:10]]
    if config.modality in ['MRI', 'MRInoram', 'CT']:
        mask = torch.stack([inp[0].unsqueeze(0) > inp[0].min() for inp in input])
        if config.get_images:
            anomaly_map *= mask
            mins = [(map[map > 0]) for map in anomaly_map]
            mins = [map.min() for map in mins]

            anomaly_map = torch.cat([(map - min) for map, min in zip(anomaly_map, mins)]).unsqueeze(1)
        anomaly_map *= mask
        anomaly_score = torch.tensor([map[inp[0].unsqueeze(0) > inp[0].min()].max()
                                      for map, inp in zip(anomaly_map, input)])

    # elif config.modality in ['RF'] and config.dataset == 'DDR':
    #     mask = torch.stack([inp.mean(0, keepdim=True) > inp.mean(0, keepdim=True).min()for inp in input])

    #     anomaly_map *= mask
    #     anomaly_score = torch.tensor([map[inp.mean(0, keepdim=True) > inp.mean(0, keepdim=True).min()].mean()
    #                                   for map, inp in zip(anomaly_map, input)])
    elif config.modality in ['RF'] and config.dataset == 'DDR':
        anomaly_score = torch.tensor([map.max() for map in anomaly_map])
    else:
        anomaly_score = torch.tensor([map.mean() for map in anomaly_map])
    if return_loss:
        return loss.item(), anomaly_map, anomaly_score
    else:
        return anomaly_map, anomaly_score  # , activations


def validate(val_loader, config):

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

    print('Starting training...')
    i_epoch = 0
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

            if config.early_validation:
                if config.step % config.anom_val_frequency == 0 or config.step == 1 or config.step == 10 or config.step == 100 or config.step == 500 or config.step == 50:
                    evaluate(config, small_testloader, val_step, val_loader)
            else:
                #
                # or config.step == 1 or config.step == 10 or config.step == 100 or config.step == 500 or config.step == 50:
                if config.step % config.anom_val_frequency == 0:

                    evaluate(config, small_testloader, val_step, val_loader)

            if config.step % config.save_frequency == 0:
                save_model(model, config, config.step)

            if config.step >= config.max_steps:
                save_model(model, config)
                print(f'Reached {config.max_steps} iterations. Finished training {config.name}.')
                return

        i_epoch += 1
       # print(f'Finished epoch {i_epoch}, ({config.step} iterations)')


if __name__ == '__main__':

    if config.eval:
        print('Evaluating model...')
        evaluate(config, big_testloader, val_step, val_loader)
        # evaluate_hist(config, big_testloader, val_step, val_loader)

    else:
        train()
