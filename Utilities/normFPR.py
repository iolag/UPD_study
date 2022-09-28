
"""
adapted from:
https://github.com/yousuhang/Unsupervised-Lesion-Detection-via-Image-Restoration-with-a-Normative-Prior
"""
from Utilities.metrics import compute_dice
from Utilities.utils import log
from torch.utils.data import DataLoader
import torch
import numpy as np
from argparse import Namespace


def dice_f1_normal_nfpr(val_loader: DataLoader,
                        config: Namespace, val_step: object = None,
                        anomaly_maps: list = None, masks: list = None,
                        anomaly_scores: list = None, labels: list = None,
                        normal_residuals: list = None, normal_scores: list = None) -> None:
    """
    Calculates and logs (wandb) Dice (pixel-level) and/or f1-score (image-level)
    on n%fpr of normal samples.

    Args:
        val_loader: validation set dataloader
        config
        anomaly_maps: model output anomaly maps
        masks: ground truth anomaly masks
        anomaly_scores:  model output anomaly scores
        labels: ground truth binary labels
        normal_residuals: model output anomaly maps on normal validation samples
        normal_scores: model output anomaly scores on normal validation samples

    """
    if normal_residuals is None:  # in the padim case these and normal_scores are given
        normal_residuals = []
        normal_scores = []

        for input in val_loader:
            input = input.to(config.device)
            # x : [anomaly_map, anomaly_score] or [anomaly_map, anomaly_score, recon]
            x = val_step(input, return_loss=False)
            normal_residuals.append(x[0].cpu())
            normal_scores.append(x[1].cpu())

    if config.dice_normal and masks is not None:

        thresh = threshold_normal_nfpr(torch.cat(normal_residuals), config.nfpr)
        dice_nfpr = compute_dice(torch.where(torch.cat(anomaly_maps) >
                                 thresh, 1, 0), torch.cat(masks))
        print(f"Dice at {config.nfpr * 100}% fpr on normal samples: {dice_nfpr:.4f}")
        log({f'anom_val/dice-norm-{config.nfpr * 100}_fpr': dice_nfpr}, config)
        return thresh
    if config.f1_normal:

        thresh = threshold_normal_nfpr(torch.cat(normal_scores), config.nfpr)
        dice_nfpr = compute_dice(torch.where(torch.cat(anomaly_scores) > thresh, 1, 0), torch.cat(labels))
        print(f"F1-score at {config.nfpr * 100}% fpr on normal samples: {dice_nfpr:.4f}, threshold: {thresh}")
        log({f'anom_val/F1-norm-{config.nfpr * 100}_fpr': dice_nfpr}, config)
        return thresh


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
