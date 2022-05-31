from Utilities.metrics import compute_dice
from torch import nn
from torch.utils.data import DataLoader
import torch
import numpy as np
from argparse import Namespace
import wandb


def dice_normal_nfpr(model: nn.Module, val_loader: DataLoader,
                     val_step: object, config: Namespace,
                     logger: wandb.run, i_iter: int,
                     anomaly_maps: list, segmentations: list) -> None:

    normal_residuals = []

    for input in val_loader:
        input = input.to(config.device)
        # x = [anomaly_map, anomaly_score] or [anomaly_map, anomaly_score, recon]
        x = val_step(model, input, return_loss=False)
        normal_residuals.append(x[0].cpu())

    thresh = threshold_normal_nfpr(torch.cat(normal_residuals), config.nfpr)
    dice_nfpr = compute_dice(torch.where(torch.cat(anomaly_maps) > thresh, 1, 0), torch.cat(segmentations))
    print(f"Dice at {config.nfpr * 100}% fpr on normal samples: {dice_nfpr:.4f}")
    logger.log({
        f'anom_val/dice-norm-{config.nfpr * 100}_fpr': dice_nfpr
    }, step=i_iter)

    return


def threshold_normal_nfpr(normal_residuals, fprate):

    normal_residuals = np.asarray(normal_residuals).reshape(-1)
    nums = len(normal_residuals)  # total num of pixels

    def func(threshold):
        # residuals after threshold (false positives)
        normal_residuals_ = normal_residuals > threshold
        # fprate = false_positives/num_samples
        fprate_ = np.sum(normal_residuals_) / np.float(nums)
        # func to minimize (l2 of current to given max fpr)
        return np.sqrt((fprate - fprate_) ** 2)

    return gss(func, normal_residuals.min(), normal_residuals.mean(), normal_residuals.max(), tau=1e-8)


def gss(f, a, b, c, tau=1e-3):
    '''
    Python recursive version of Golden Section Search algorithm
    tau is the tolerance for the minimal value of function f
    b is any number between the interval a and c
    '''
    goldenRatio = (1 + 5 ** 0.5) / 2

    if (c - b > b - a):
        x = b + (2 - goldenRatio) * (c - b)
    else:
        x = b - (2 - goldenRatio) * (b - a)

    # termination condition
    if (abs(c - a) < tau * (abs(b) + abs(x))):
        return (c + a) / 2

    if (f(x) < f(b)):
        if (c - b > b - a):
            return gss(f, b, x, c, tau)
        return gss(f, a, x, b, tau)
    else:
        if (c - b > b - a):
            return gss(f, a, b, x, tau)
        return gss(f, x, b, c, tau)
