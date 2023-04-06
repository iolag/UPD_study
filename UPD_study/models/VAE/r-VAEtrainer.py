"""
adapted for PyTorch from:
https://github.com/yousuhang/Unsupervised-Lesion-Detection-via-Image-Restoration-with-a-Normative-Prior
"""
"""
Note that this file includes only the evaluation logic for r-Vae. 
For it to work, a VAE should have been trained first, with the same modality, model hyperparameters and seed. 
Then the appropriate model will then be loaded automatically when you run this script. 
"""

from argparse import ArgumentParser
import numpy as np
import torch
from tqdm import tqdm
from VAEmodel import VAE
from torch import Tensor
from typing import Tuple
from torchinfo import summary
from scipy.ndimage import gaussian_filter
import pathlib
from UPD_study.utilities.evaluate import evaluate
from UPD_study.utilities.common_config import common_config
from UPD_study.utilities.utils import (seed_everything, load_data, load_pretrained,
                                       misc_settings, ssim_map, load_model, test_inference_speed)

""""""""""""""""""""""""""""""""""" Config """""""""""""""""""""""""""""""""""


def get_config():
    parser = ArgumentParser()
    parser = common_config(parser)

    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_restoration_steps', type=int, default=500, help='Number of restoration steps.')
    parser.add_argument('--restore_lr', type=float, default=1e3, help='Restoration learning rate.')
    parser.add_argument('--tv_lambda', type=float, default=-1, help='Total variation weight, lambda.')
    parser.add_argument('--kl_weight', type=float, default=0.001, help='kl loss term weight')
    parser.add_argument('--latent_dim', type=int, default=512, help='Latent dimension.')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Numer of encoder (and decoder) conv layers')
    parser.add_argument('--width', type=int, default=16, help='First conv layer number of filters.')
    parser.add_argument('--conv1x1', type=int, default=16,
                        help='Channel downsampling with 1x1 convs before bottleneck.')
    parser.add_argument('--kernel_size', type=int, default=3, help='Convolutional kernel size.')
    parser.add_argument('--padding', type=int, default=1,
                        help='Padding for consistent downsampling, set to 2 if kernel_size == 5.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Input Dropout like https://doi.org/10.1145/1390156.1390294')
    return parser.parse_args()


config = get_config()

# set initial script settings
config.restoration = True
config.method = 'VAE'
config.eval = True

config.model_dir_path = pathlib.Path(__file__).parents[0]
misc_settings(config)

# Specific modality params (Default are for MRI t2)
if config.modality != 'MRI':
    config.max_steps = 10000

if config.modality == 'CXR':
    config.kl_weight = 0.0001
    config.num_layers = 6
    config.latent_dim = 256
    config.width = 16
    config.conv1x1 = 64

if (config.modality == 'MRI' and config.sequence == 't1') or config.modality == 'RF':
    config.kl_weight = 0.0001
    config.num_layers = 6
    config.latent_dim = 512
    config.width = 32
    config.conv1x1 = 64

# lamdas pre-calculated for all datasets
if config.modality == 'CXR':
    if config.load_pretrained:
        config.tv_lambda = 1.3
    else:
        config.tv_lambda = 1.5

if config.modality == 'MRI':
    if config.load_pretrained:
        config.tv_lambda = 1.9
    else:
        config.tv_lambda = 1.2

if config.modality == 'RF':
    config.tv_lambda = 1.7


""""""""""""""""""""""""""""""""" Load data """""""""""""""""""""""""""""""""

# specific seed for creating the dataloader
seed_everything(42)
_, _, big_testloader, small_testloader = load_data(config)

if config.tv_lambda < 0:
    # to calculate tv_lambda set config.eval = False and get normal validation set
    config.eval = False

seed_everything(42)
_, val_loader, _, _ = load_data(config)

""""""""""""""""""""""""""""""""" Init model """""""""""""""""""""""""""""""""
# Reproducibility
seed_everything(config.seed)

print("Initializing model...")
model = VAE(config).to(config.device)

# load CCD pretrainerd encoder and bottleneck
if config.load_pretrained and not config.eval:
    config.arch = 'vae'
    model = load_pretrained(model, config)

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


def total_variation(img):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)


def determine_best_lambda():
    """
    Calculate best lambda weight of total variation loss term
    """
    lambdas = np.arange(20) / 10.0
    mean_errors = []
    for tv_lambda in tqdm(lambdas, desc='Calculating optimal lamda'):
        errors = []
        for input in val_loader:
            input = input.to(config.device)
            input.requires_grad = True
            restored = input.clone()
            for step in range(config.num_restoration_steps):
                reconstruction, mu, logvar = model(restored)
                tv_loss = tv_lambda * total_variation(restored - input)
                # get ELBO(restoration)
                elbo = model.loss_function(restored, reconstruction, mu, logvar)['loss']
                grad = torch.autograd.grad((tv_loss + elbo), restored, retain_graph=True)[0]
                grad = torch.clamp(grad, -50., 50.)
                restored -= config.restore_lr * grad
            errors.append(torch.abs(input - restored).mean())
        mean_error = torch.mean(torch.tensor(errors))
        mean_errors.append(mean_error)

    config.tv_lambda = lambdas[mean_errors.index(min(mean_errors))]
    print(f'Best lambda: { config.tv_lambda}')


def restore(input):
    """
    Iteratively restore input
    """
    input = input.to(config.device)
    input.requires_grad = True
    restored = input.clone()
    for step in range(config.num_restoration_steps):
        reconstruction, mu, logvar = model(restored)
        tv_loss = config.tv_lambda * total_variation(restored - input)
        elbo = model.loss_function(restored, reconstruction, mu, logvar)["loss"]
        grad = torch.autograd.grad((tv_loss + elbo), restored, retain_graph=True)[0]
        grad = torch.clamp(grad, -50., 50.)
        # if step % 100 == 0:
        #     print(tv_loss.mean().item(), 'tv_loss', elbo.mean().item(), 'elbo loss')

        restored -= config.restore_lr * grad
    return restored


def restoration_step(input, test_samples: bool = False) -> Tuple[dict, Tensor]:
    """
    Iteratively restore input
    """
    model.eval()
    input_recon = restore(input)

    # Anomaly map
    if config.ssim_eval:
        anomaly_map = ssim_map(input_recon, input)

        if config.gaussian_blur:
            anomaly_map = anomaly_map.detach().cpu().numpy()
            for i in range(anomaly_map.shape[0]):
                anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)
            anomaly_map = torch.from_numpy(anomaly_map).to(config.device)
    else:
        anomaly_map = (input - input_recon).abs().mean(1, keepdim=True).detach()
        if config.gaussian_blur:
            anomaly_map = anomaly_map.cpu().numpy()
            for i in range(anomaly_map.shape[0]):
                anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)
            anomaly_map = torch.from_numpy(anomaly_map).to(config.device)

    # for MRI, RF apply brainmask
    if config.modality == 'MRI':
        mask = torch.stack([inp > inp.min() for inp in input])
        anomaly_map *= mask
        input_recon *= mask
        anomaly_score = torch.tensor([map[inp > inp.min()].max() for map, inp in zip(anomaly_map, input)])

    elif config.modality == 'RF':
        anomaly_score = torch.tensor([map.max() for map in anomaly_map])
    else:
        anomaly_score = torch.tensor([map.mean() for map in anomaly_map])

    return anomaly_map.detach(), anomaly_score.detach(), input_recon.detach()


if __name__ == '__main__':
    if config.speed_benchmark:
        test_inference_speed(restoration_step, iterations=100, restoration=True)
        exit(0)

    print('Evaluating model...')

    if config.tv_lambda < 0:
        determine_best_lambda()

    evaluate(config, big_testloader, restoration_step)
