from typing import Callable
import torch
from UPD_study.utilities.utils import metrics, log
from argparse import Namespace
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate(config: Namespace, test_loader: DataLoader, val_step: Callable) -> None:
    """
    Common evaluation method. Handles inference on evaluation set, metric calculation,
    logging and the speed benchmark.

    Args:
        config (Namespace): configuration object.
        test_loader (DataLoader): evaluation set dataloader
        val_step (Callable): validation step function
    """

    labels = []
    anomaly_scores = []
    anomaly_maps = []
    inputs = []
    segmentations = []

    # forward pass the testloader to extract anomaly maps, scores, masks, labels
    for input, mask in tqdm(test_loader, desc="Test set", disable=config.speed_benchmark):
        input = input.to(config.device)
        output = val_step(input, test_samples=True)

        anomaly_map, anomaly_score = output[:2]
        inputs.append(input.cpu())

        if config.method == 'Cutpaste' and config.localization:
            anomaly_maps.append(anomaly_map.cpu())
            segmentations.append(mask)

            label = torch.where(mask.sum(dim=(1, 2, 3)) > 0, 1, 0)
            labels.append(label)
            anomaly_scores.append(torch.zeros_like(label))
        elif config.method == 'Cutpaste' and not config.localization:
            segmentations = None
            anomaly_scores.append(anomaly_score.cpu())
            label = torch.where(mask.sum(dim=(1, 2, 3)) > 0, 1, 0)
            labels.append(label)
        else:
            anomaly_maps.append(anomaly_map.cpu())
            segmentations.append(mask)
            anomaly_scores.append(anomaly_score.cpu())
            label = torch.where(mask.sum(dim=(1, 2, 3)) > 0, 1, 0)
            labels.append(label)

    metrics(config, anomaly_maps, segmentations, anomaly_scores, labels)

    # do a single forward pass to extract images to log
    # the batch size is num_images_log for test_loaders, so only a single forward pass necessary
    input, mask = next(iter(test_loader))
    output = val_step(input.to(config.device), test_samples=True)

    anomaly_maps = output[0]

    log({'anom_val/input images': input,
        'anom_val/targets': mask,
         'anom_val/anomaly maps': anomaly_maps}, config)

    # if recon based method, len(x)==3 because val_step returns reconstructions.
    # if thats the case log the reconstructions
    if len(output) == 3:
        log({'anom_val/reconstructions': output[2]}, config)
