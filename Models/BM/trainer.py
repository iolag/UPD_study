"""
Implementation of the Baseline Model from:
'Simple statistical methods for unsupervised brain anomaly detection
on MRI are competitive to deep learning methods'
https://arxiv.org/pdf/2011.12735.pdf
"""
import sys
sys.path.append('/data_ssd/users/lagi/thesis/UAD_study/')
import argparse
from typing import Tuple
import numpy as np
import torch
from DatasetPreprocessing.mri import get_camcan_slices, get_brats_slices, get_atlas_slices
from Utilities.utils import metrics, seed_everything


""" Config """
parser = argparse.ArgumentParser()

# General settings
parser.add_argument("--seed", type=int, default=42, help='Random seed')

# Data settings
parser.add_argument('--datasets_dir', type=str,
                    default='/datasets/Datasets/', help='datasets_dir')
parser.add_argument('--image_size', type=int, default=128, help='Image size')
parser.add_argument('--sequence', type=str, default='t2', help='MRI sequence')
parser.add_argument('--slice_range', type=int, nargs='+',
                    default=(0, 155), help='Lower and Upper slice index')
parser.add_argument('--normalize', type=bool, default=False, help='Normalize images between 0 and 1')
parser.add_argument('--equalize_histogram', type=bool, default=False, help='Equalize histogram')


config = parser.parse_args()


seed_everything(config.seed)


def train(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the parameters for the distance (mu and std).
    :param x: Training data (shape: [n, c, h, w])
    """
    # Mean
    mu = x.mean(axis=0)

    # Standard deviation
    std = np.std(x, axis=0, ddof=1)  # ddof=1 to get unbiased std (divide by N-1)
    std[std == 0] = 1  # Avoid division by zero for background pixels

    return mu, std


def predict(x: np.ndarray, mu: np.ndarray, std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict the anomaly score for the test data.
    :param x: Test data (shape: [n, c, h, w])
    :param mu: Mean of the training data
    :param std: Standard deviation of the training data
    """
    # Normalize the test data
    anomaly_maps = np.absolute(x - mu) / std
    # apply masks
    masks = np.stack([inp > inp.min() for inp in x])
    anomaly_maps *= masks

    # Compute the anomaly score
    anomaly_scores = np.asarray([map[inp > inp.min()].mean() for map, inp in zip(anomaly_maps, x)])

    return anomaly_maps, anomaly_scores


if __name__ == '__main__':
    # Load the data
    print("Loading data...")
    config.return_volumes = True
    train_images = get_camcan_slices(config)

    # get mean, std of one volume
    volume = train_images[0]
    mean = np.mean(volume)  # [volume > 0])
    std = np.std(volume)  # [volume > 0])

    # normalize train images
    # train_images = np.concatenate((train_images - mean) / std)
    train_images = np.concatenate([(volume - np.mean(volume)) / np.std(volume) for volume in train_images])
    # masks = np.stack([inp > inp.min() for inp in train_images])
    # train_images *= masks

    # remove slices with no brain pixels in them
    train_images = train_images[np.sum(train_images, axis=(1, 2, 3)) > 0]

    print(f"train_images.shape: {train_images.shape}")

    if config.sequence == 't1':
        test_images, segmentations = get_atlas_slices(config)
    if config.sequence == 't2':
        test_images, segmentations = get_brats_slices(config)

    # normalize test images
    # test_images = np.concatenate((test_images - mean) / std)
    test_images = np.concatenate([(volume - np.mean(volume)) / np.std(volume) for volume in test_images])
    segmentations = np.concatenate(segmentations)
    # remove slices with no brain pixels in them
    non_zero_idx = np.sum(test_images, axis=(1, 2, 3)) > 0
    test_images = test_images[non_zero_idx]
    segmentations = segmentations[non_zero_idx]

    labels = np.where(segmentations.sum(axis=(1, 2, 3)) > 0, 1, 0)
    print(f"test_images.shape: {test_images.shape}")

    # Train the model
    print("Estimating the parameters...")
    mu, std = train(train_images)

    # Predict the anomaly score
    print("Predicting the anomaly maps and scores...")
    anomaly_maps, anomaly_scores = predict(test_images, mu, std)

    # make it compatible with metrics(), which expects lists of torch.Tensor batches (shape: [b,c,h,w])
    anomaly_maps = torch.unbind(torch.from_numpy(anomaly_maps))
    segmentations = torch.unbind(torch.from_numpy(segmentations))
    anomaly_scores = torch.unbind(torch.from_numpy(anomaly_scores).unsqueeze(1))
    labels = torch.unbind(torch.from_numpy(labels).unsqueeze(1))

    # Evaluate the model
    print("Evaluating...")
    metrics(anomaly_maps, segmentations, anomaly_scores, labels, limited_metrics=True)
