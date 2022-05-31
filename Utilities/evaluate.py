import sys
import os
sys.path.append('/home/ioannis/lagi/thesis/UAD_study')
from torch import nn
import torch
from Utilities.utils import metrics
import wandb
os.environ["WANDB_SILENT"] = "true"
from argparse import Namespace
from torch.utils.data import DataLoader
from Utilities.dice_normfpr import dice_normal_nfpr
from tqdm import tqdm


def evaluate(model: nn.Module, test_loader: DataLoader,
             i_iter: int, val_step: object, logger: wandb.run,
             config: Namespace, val_loader: DataLoader = None) -> None:
    """
    because we work we datasets of both with and without masks, code is structured in a
    way that no mask dataloaders also return masks. For healthy samples they return empty
    masks, and for anomalous samples they return non-empty masks. The code then works the
    same and the binary label is produced by label = torch.where(mask.sum(dim=(1, 2, 3)) > 0, 1, 0)
    """
    labels = []
    anomaly_scores = []
    anomaly_maps = []
    segmentations = []

    # forward pass the testloader to extract anomaly maps, scores, masks, labels
    for input, mask in tqdm(test_loader, desc="Test set"):

        # forward pass, x = [anomaly_map, anomaly_score] or [anomaly_map, anomaly_score, recon]
        x = val_step(model, input.to(config.device), return_loss=False)

        anomaly_maps.append(x[0].cpu())
        anomaly_scores.append(x[1].cpu())
        segmentations.append(mask)
        label = torch.where(mask.sum(dim=(1, 2, 3)) > 0, 1, 0)
        labels.append(label)

    # calculate metrics like AP, AUROC, on pixel and/or image level
    if config.norm_fpr:
        config.nfpr = 0.05
        dice_normal_nfpr(model, val_loader, val_step, config, logger, i_iter, anomaly_maps, segmentations)
        config.nfpr = 0.1
        dice_normal_nfpr(model, val_loader, val_step, config, logger, i_iter, anomaly_maps, segmentations)

    if config.modality in ['CXR', 'OCT']:
        segmentations = None  # disables pixel level evaluation in metrics()

    threshold = metrics(anomaly_maps, segmentations, anomaly_scores,
                        labels, logger, i_iter, config.limited_metrics)

    # do a single forward pass to extract stuff to log
    # the batch size is num_images_log for test_loaders, so a single forward pass is enough
    input, mask = next(iter(test_loader))

    # forward pass, x = [anomaly_map, anomaly_score] or [anomaly_map, anomaly_score, recon]
    x = val_step(model, input.to(config.device), return_loss=False)

    # Log images to wandb
    input_images = list(input[:config.num_images_log].cpu())
    targets = list(mask.float()[:config.num_images_log].cpu())

    anomaly_images = list(x[0][:config.num_images_log].cpu())
    logger.log({
        'anom_val/input images': [wandb.Image(img) for img in input_images],
        'anom_val/targets': [wandb.Image(img) for img in targets],
        'anom_val/anomaly maps': [wandb.Image(img) for img in anomaly_images]
    }, step=i_iter)

    # if recon based method, len(x)==3 because val_step returns reconstructions.
    # if thats the case  log the reconstructions
    if len(x) == 3:
        input_reconstructions = list(x[2][:config.num_images_log].cpu())
        logger.log({
            'anom_val/reconstructions': [wandb.Image(img) for img in input_reconstructions],
        }, step=i_iter)

    # produce thresholded images
    if not config.limited_metrics:
        # create thresholded images on the threshold of best posible dice score on the dataset
        anomaly_thresh_map = x[0][:config.num_images_log]
        anomaly_thresh_map = torch.where(anomaly_thresh_map < threshold,
                                         torch.FloatTensor([0.]).to(config.device),
                                         torch.FloatTensor([1.]).to(config.device))
        anomaly_thresh = list(anomaly_thresh_map.cpu())

        logger.log({
            'anom_val/thresholded maps': [wandb.Image(img) for img in anomaly_thresh],
        }, step=i_iter)
