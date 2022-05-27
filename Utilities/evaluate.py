
import sys
import os
sys.path.append('/u/home/lagi/thesis')

import torch
from Utilities.utils import metrics
import wandb
os.environ["WANDB_SILENT"] = "true"


def eval_reconstruction_based(model, dataloader, i_iter, val_step, logger, config) -> None:

    labels = []
    anomaly_scores = []
    anomaly_maps = []
    segmentations = []

    for input, mask in dataloader:  # input, mask: [b, c, h, w]
        # Compute anomaly map
        input = input.to(config.device)
        _, anomaly_map, anomaly_score, input_recon = val_step(model, input, return_loss=False)

        anomaly_maps.append(anomaly_map.cpu())
        anomaly_scores.append(anomaly_score.cpu())

        segmentations.append(mask)
        label = torch.where(mask.sum(dim=(1, 2, 3)) > 0, 1, 0)
        labels.append(label)

    # calculate metrics like AP, AUROC, on pixel and/or image level
    if config.eval:
        threshold = metrics(anomaly_maps, segmentations, anomaly_scores,
                            labels, logger, i_iter, limited_metrics=False)
    else:
        threshold = metrics(anomaly_maps, segmentations, anomaly_scores,
                            labels, logger, i_iter, limited_metrics=True)

    for input, mask in dataloader:  # input, mask: [b, c, h, w]

        # Compute anomaly map
        input = input.to(config.device)
        _, anomaly_map, anomaly_score, input_recon = val_step(model, input)

        # Log images to wandb
        input_images = list(input[:config.num_images_log].cpu())
        targets = list(mask.float()[:config.num_images_log].cpu())
        input_reconstructions = list(input_recon[:config.num_images_log].cpu())
        anomaly_images = list(anomaly_map[:config.num_images_log].cpu())

        # create thresholded images on the threshold of best posible dice score on the dataset
        anomaly_thresh_map = anomaly_map[:config.num_images_log]
        anomaly_thresh_map = torch.where(anomaly_thresh_map < threshold,
                                         torch.FloatTensor([0.]).to(config.device),
                                         torch.FloatTensor([1.]).to(config.device))

        anomaly_thresh = list(anomaly_thresh_map.cpu())

        logger.log({
            'anom_val/input images': [wandb.Image(img) for img in input_images],
            'anom_val/targets': [wandb.Image(img) for img in targets],
            'anom_val/reconstructions': [wandb.Image(img) for img in input_reconstructions],
            'anom_val/anomaly maps': [wandb.Image(img) for img in anomaly_images],
            'anom_val/thresholded maps': [wandb.Image(img) for img in anomaly_thresh],
        }, step=i_iter)
        break


def eval_dfr_pii(model, dataloader, i_iter, val_step, logger, config):

    labels = []
    anomaly_scores = []
    anomaly_maps = []
    segmentations = []

    for input, mask in dataloader:  # input, mask: [b, c, h, w]
        input = input.to(config.device)
        # Compute loss, anomaly map and anomaly score
        anomaly_map, anomaly_score = val_step(model, input, return_loss=False)

        anomaly_maps.append(anomaly_map.cpu())
        anomaly_scores.append(anomaly_score.cpu())
        segmentations.append(mask)

        label = torch.where(mask.sum(dim=(1, 2, 3)) > 0, 1, 0)
        labels.append(label)

    if config.eval:
        threshold = metrics(anomaly_maps, segmentations, anomaly_scores,
                            labels, logger, i_iter, limited_metrics=False)
    else:
        threshold = metrics(anomaly_maps, segmentations, anomaly_scores, labels, logger, i_iter)

    i = 0
    for input, mask in dataloader:  # input, mask: [b, c, h, w]
        i += 1
        input = input.to(config.device)
        # Compute loss, anomaly map and anomaly score
        _, anomaly_map, anomaly_score = val_step(model, input)

        # Log images to wandb
        input_images = list(input[:config.num_images_log].cpu())
        targets = list(mask.float()[:config.num_images_log].cpu())
        anomaly_images = list(anomaly_map[:config.num_images_log].cpu())

        # create thresholded images on the threshold of best posible dice score on the dataset
        anomaly_thresh_map = anomaly_map[:config.num_images_log]
        anomaly_thresh_map = torch.where(anomaly_thresh_map < threshold,
                                         torch.FloatTensor([0.]).to(config.device),
                                         torch.FloatTensor([1.]).to(config.device))
        anomaly_thresh = list(anomaly_thresh_map.cpu())

        logger.log({
            'anom_val/input images': [wandb.Image(img) for img in input_images],
            'anom_val/targets': [wandb.Image(img) for img in targets],
            'anom_val/anomaly maps': [wandb.Image(img) for img in anomaly_images],
            'anom_val/thresholded maps': [wandb.Image(img) for img in anomaly_thresh],
        }, step=i)
        if i == 5:
            break