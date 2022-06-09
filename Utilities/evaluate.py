import sys
sys.path.append('/data_ssd/users/lagi/thesis/UAD_study/')
import torch
from Utilities.utils import metrics, log
from argparse import Namespace
from torch.utils.data import DataLoader
from Utilities.normFPR import dice_f1_normal_nfpr
from tqdm import tqdm


def evaluate(config: Namespace, test_loader: DataLoader,
             val_step: object, val_loader: DataLoader = None) -> None:
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

        if config.patches:
            # input.shape = [samples,crops,c,h,w]
            # mask.shape = [samples,crops,1,h,w]
            b, num_crops, _, _, _ = input.shape
            input = input.view(input.shape[0] * input.shape[1], *input.shape[2:])  # [samples*crops,c,h,w]
            #  print(input.shape)
            # mask = mask.reshape(mask.shape[0] * mask.shape[1], *mask.shape[2:])# [samples*crops,1,h,w]
            # print(mask.shape)
            x = val_step(input.to(config.device), return_loss=False)
            # print(x[1])
            per_crop_scores = x[1].view(b, config.num_patches)  # [samples,crops]
            # print(per_crop_scores.shape)
            import math

            anomaly_score = per_crop_scores.mean(1)  # [samples]
            discard_nan = False
            for i in anomaly_score:

                if math.isnan(i):
                    discard_nan = True
            if not discard_nan:
                # print(anomaly_score.shape)
                anomaly_scores.append(anomaly_score.cpu())
                # [samples, crops] crop-wise binary vector
                crop_label = torch.where(mask.sum(dim=(2, 3, 4)) > 0, 1, 0)  # [samples, crops]
                # print(crop_label.shape)
                label = torch.where(crop_label.sum(dim=1) > 0, 1, 0)  # [samples] sample-wise binary vector
                # print(label.shape)
                labels.append(label)
                if config.dataset == 'IDRID':

                    map = x[0].view(b, 1, config.image_size * int(math.sqrt(config.num_patches)),
                                    config.image_size * int(math.sqrt(config.num_patches)))
                    anomaly_maps.append(map.cpu())
                    segmentations.append(mask)
        else:
            # forward pass, x = [anomaly_map, anomaly_score] or [anomaly_map, anomaly_score, recon]
            x = val_step(input.to(config.device), return_loss=False)

            anomaly_maps.append(x[0].cpu())
            anomaly_scores.append(x[1].cpu())
            segmentations.append(mask)
            label = torch.where(mask.sum(dim=(1, 2, 3)) > 0, 1, 0)
            labels.append(label)

    # calculate metrics like AP, AUROC, on pixel and/or image level

    if config.modality in ['CXR', 'OCT'] or (config.dataset != 'IDRID' and config.modality == 'RF'):
        segmentations = None  # disables pixel level evaluation in metrics()

    threshold = metrics(config, anomaly_maps, segmentations, anomaly_scores, labels)

    if config.normal_fpr:
        config.nfpr = 0.05
        dice_f1_normal_nfpr(val_loader, config, val_step,
                            anomaly_maps, segmentations, anomaly_scores, labels)
        config.nfpr = 0.1
        dice_f1_normal_nfpr(val_loader, config, val_step, anomaly_maps,
                            segmentations, anomaly_scores, labels)

    # if config.patches:
    #     return

    # do a single forward pass to extract stuff to log
    # the batch size is num_images_log for test_loaders, so a single forward pass is enough
    input, mask = next(iter(test_loader))

    if config.patches:
        input = input.view(input.shape[0] * input.shape[1], *input.shape[2:])
    # forward pass, x = [anomaly_map, anomaly_score] or [anomaly_map, anomaly_score, recon]
    x = val_step(input.to(config.device), return_loss=False)

    if config.patches:
        input = input.view(4, config.img_channels, config.image_size * int(math.sqrt(config.num_patches)),
                           config.image_size * int(math.sqrt(config.num_patches)))
        map = x[0].view(4, 1, config.image_size * int(math.sqrt(config.num_patches)),
                        config.image_size * int(math.sqrt(config.num_patches)))

        mask = mask.squeeze(1)

    else:
        maps = x[0]

    log({'anom_val/input images': input,
         'anom_val/targets': mask,
         'anom_val/anomaly maps': maps}, config)
    # # Log images to wandb
    # input_images = list(input[:config.num_images_log].cpu())
    # targets = list(mask.float()[:config.num_images_log].cpu())

    # anomaly_images = list(maps[:config.num_images_log].cpu())
    # config.logger.log({
    #     'anom_val/input images': [wandb.Image(img) for img in input_images],
    #     'anom_val/targets': [wandb.Image(img) for img in targets],
    #     'anom_val/anomaly maps': [wandb.Image(img) for img in anomaly_images]
    # }, step=config.step)

    # if recon based method, len(x)==3 because val_step returns reconstructions.
    # if thats the case  log the reconstructions
    if len(x) == 3:

        log({'anom_val/reconstructions': x[2]}, config)
    # produce thresholded images
    if not config.limited_metrics and segmentations is not None:
        # create thresholded images on the threshold of best posible dice score on the dataset
        anomaly_thresh_map = torch.where(x[0] < threshold,
                                         torch.FloatTensor([0.]).to(config.device),
                                         torch.FloatTensor([1.]).to(config.device))

        log({'anom_val/thresholded maps': anomaly_thresh_map}, config)
