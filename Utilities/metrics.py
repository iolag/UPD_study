from functools import partial
from multiprocessing import Pool
from typing import Tuple
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve  # ,auc


def compute_average_precision(predictions, targets):
    """
    Compute Average Precision
    Args:
        predictions (torch.Tensor): Anomaly scores
        targets (torch.Tensor): Segmentation map or target label, must be binary
    """
    if (targets - targets.int()).sum() > 0.:
        raise RuntimeError("targets for AP must be binary")
    ap = average_precision_score(targets.reshape(-1), predictions.reshape(-1))
    return ap


def compute_auroc(predictions, targets) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic Curve
    Args:
        predictions (torch.Tensor): Anomaly scores
        targets (torch.Tensor): Segmentation map or target label, must be binary
    """
    if (targets - targets.int()).sum() > 0.:
        raise RuntimeError("targets for AUROC must be binary")
    auc = roc_auc_score(targets.reshape(-1), predictions.reshape(-1))
    return auc


def compute_dice(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Computes the Sorensen-Dice coefficient:

    dice = 2 * TP / (2 * TP + FP + FN)

    :param preds: An array of predicted anomaly scores.
    :param targets: An array of ground truth labels.
    """
    preds, targets = np.array(preds), np.array(targets)

    # Check if predictions and targets are binary
    if not np.all(np.logical_or(preds == 0, preds == 1)):
        raise ValueError('Predictions must be binary')
    if not np.all(np.logical_or(targets == 0, targets == 1)):
        raise ValueError('Targets must be binary')

    # Compute Dice
    dice = 2 * np.sum(preds[targets == 1]) / \
        (np.sum(preds) + np.sum(targets))

    return dice


def compute_dice_at_nfpr(preds: np.ndarray, targets: np.ndarray,
                         max_fpr: float = 0.05) -> float:
    """
    Computes the Sorensen-Dice score at 5% FPR.

    :param preds: An array of predicted anomaly scores.
    :param targets: An array of ground truth labels.
    :param max_fpr: Maximum false positive rate.
    :param n_threshs: Maximum number of thresholds to check.
    """
    preds, targets = np.array(preds), np.array(targets)

    # Find threshold for 5% FPR
    fpr, _, thresholds = roc_curve(targets.reshape(-1), preds.reshape(-1))
    t = thresholds[max(0, fpr.searchsorted(max_fpr, 'right') - 1)]

    # Compute Dice
    return compute_dice(np.where(preds > t, 1, 0), targets)


def compute_thresh_at_nfpr(preds: np.ndarray, targets: np.ndarray,
                           max_fpr: float = 0.05) -> float:
    """
    Computes the threshold at 5% FPR.

    :param preds: An array of predicted anomaly scores.
    :param targets: An array of ground truth labels.
    :param max_fpr: Maximum false positive rate.
    """
    preds, targets = np.array(preds), np.array(targets)

    # Find threshold for 5% FPR
    fpr, _, thresholds = roc_curve(targets.reshape(-1), preds.reshape(-1))
    t = thresholds[max(0, fpr.searchsorted(max_fpr, 'right') - 1)]

    # Return threshold
    return t


def compute_best_dice(preds: np.ndarray, targets: np.ndarray,
                      n_thresh: float = 100,
                      num_processes: int = 4) -> Tuple[float, float]:
    """
    Compute the best dice score for n_thresh thresholds.

    :param predictions: An array of predicted anomaly scores.
    :param targets: An array of ground truth labels.
    :param n_thresh: Number of thresholds to check.
    """
    preds, targets = np.array(preds), np.array(targets)

    thresholds = np.linspace(preds.max(), preds.min(), n_thresh)

    with Pool(num_processes) as pool:
        fn = partial(_dice_multiprocessing, preds, targets)
        scores = pool.map(fn, thresholds)

    scores = np.stack(scores, 0)
    max_dice = scores.max()
    max_thresh = thresholds[scores.argmax()]
    return max_dice, max_thresh


def _dice_multiprocessing(preds: np.ndarray, targets: np.ndarray,
                          threshold: float) -> float:
    return compute_dice(np.where(preds > threshold, 1, 0), targets)
