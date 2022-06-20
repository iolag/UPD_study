
import sys
import os
sys.path.append('/data_ssd/users/lagi/thesis/UAD_study/')
import numpy as np
import torch
from torchvision import transforms as T
from Utilities.metrics import (
    compute_average_precision,
    compute_auroc, compute_dice_at_nfpr,
    compute_best_dice
)
from time import time
from argparse import Namespace
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Union
from torch import Tensor
import random
import wandb
os.environ["WANDB_SILENT"] = "true"
import torchgeometry as tgm
from torch import nn


def load_pretrained(model, config):

    # use this str to load pretrained backbone
    pretrained = f'CCD_{config.arch}_{config.modality}'
    if config.modality == 'MRI':
        pretrained += f'_{config.sequence}'
    pretrained += f'_{config.name_add}'

    if config.arch == 'vae':
        enc_dict = model.encoder.state_dict()
        pretrained_dict = torch.load(
            f'/u/home/lagi/thesis/UAD_study/Models/CCD/saved_models/{config.modality}/{pretrained}_encoder_.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in enc_dict}
        enc_dict.update(pretrained_dict)
        model.encoder.load_state_dict(enc_dict)
        bn_dict = model.bottleneck.state_dict()
        pretrained_dict = torch.load(
            f'/u/home/lagi/thesis/UAD_study/Models/CCD/saved_models/{config.modality}/{pretrained}_bottleneck_.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in bn_dict}
        bn_dict.update(pretrained_dict)
        model.bottleneck.load_state_dict(bn_dict)
        print('Pretrained backbone loaded.')

    else:
        if config.method == 'f-anoGAN':
            model.fc = nn.Linear(4 * 4 * 8 * config.dim, 1024).to(config.device)

        model_dict = model.state_dict()
        pretrained_dict = torch.load(
            f'/u/home/lagi/thesis/UAD_study/Models/CCD/saved_models/{config.modality}/{pretrained}_.pth')

        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(f'weights for {k} not in pretrained weight state_dict()')

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        if config.method == 'f-anoGAN':
            model.fc = nn.Linear(4 * 4 * 8 * config.dim, config.latent_dim).to(config.device)

        print('Pretrained backbone loaded.')
    return model


def save_model(model, config, i_iter: Union[int, str] = ""):
    if config.method == 'RD':
        torch.save(model['decoder'].state_dict(),
                   f'saved_models/{config.modality}/{config.name}_dec_{i_iter}.pth')
        torch.save(model['bn'].state_dict(),
                   f'saved_models/{config.modality}/{config.name}_bn_{i_iter}.pth')

    elif config.method == 'CCD' and config.backbone_arch == 'vae':
        torch.save(model['encoder'].state_dict(),
                   f'saved_models/{config.modality}/{config.name}_encoder_{i_iter}.pth')
        torch.save(model['bottleneck'].state_dict(),
                   f'saved_models/{config.modality}/{config.name}_bottleneck_{i_iter}.pth')

    else:
        torch.save(model.state_dict(), f'saved_models/{config.modality}/{config.name}_{i_iter}.pth')


def load_model(config):

    if config.method == 'RD':

        load_dec = torch.load(f'saved_models/{config.modality}/{config.name}_dec_{config.load_iter}.pth')
        load_bn = torch.load(f'saved_models/{config.modality}/{config.name}_bn_{config.load_iter}.pth')
        return load_dec, load_bn

    elif config.method == 'f-anoGAN':

        if config.eval:
            load_g = torch.load(f'saved_models/{config.modality}/{config.name}_netG.pth'.replace('_CCD', ''))
            load_d = torch.load(f'saved_models/{config.modality}/{config.name}_netD.pth'.replace('_CCD', ''))
            load_e = torch.load(f'saved_models/{config.modality}/{config.name}_netE.pth')
            return load_g, load_d, load_e
        # the 2 distinct run scenario where wgan is already trained and we want to train encoder
        else:
            load_g = torch.load(f'saved_models/{config.modality}/{config.name}_netG.pth'.replace('_CCD', ''))
            load_d = torch.load(f'saved_models/{config.modality}/{config.name}_netD.pth'.replace('_CCD', ''))
            return load_g, load_d

    else:
        return torch.load(f'saved_models/{config.modality}/{config.name}_{config.load_iter}.pth')


def set_requires_grad(model, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad = requires_grad


def misc_settings(config):
    """
    """
    # msg = "num_images_log should be lower or equal to batch size"
    # assert (config.batch_si   ze >= config.num_images_log), msg
    if config.modality == 'RF':
        config.stadardize = True

    if not config.limited_metrics:
        config.f1_normal = True
        config.dice_normal = True
        config.normal_fpr = True

        # Select training device
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Training on {config.device}")

    # Multi purpose model name string
    if config.method == 'CCD':
        name = f'CCD_{config.backbone_arch}_{config.modality}'
    else:
        name = f'{config.method}_{config.modality}'

    if config.modality == 'MRI':
        name += f'_{config.sequence}'

    if config.load_pretrained:
        name += '_CCD'

    name += f'_{config.name_add}'

    # create saved_models folder
    os.makedirs(f'saved_models/{config.modality}', exist_ok=True)

    # init wandb logger
    if not config.eval and not config.disable_wandb:
        logger = wandb.init(project=config.method, name=name, config=config, reinit=True)
    if config.eval and not config.disable_wandb:
        logger = wandb.init(project=config.method, name=f'{name}_eval', config=config, reinit=True)
    if config.disable_wandb:
        logger = wandb.init(mode="disabled")

    # keep name, logger, step in config to pass them around
    config.name = name
    config.logger = logger
    config.step = 0

    return


# initialize SSIM Loss module
ssim_module = tgm.losses.SSIM(11, max_val=255)


def ssim_map(batch1: Tensor, batch2: Tensor):
    """
    Computes the anomaly map between two batches using SSIM.
    The torchgeometry.losses.SSIM module returns Structural Dissimilarity:

    DSSIM = (1 - SSIM)/2

    which we turn to an anomaly map. If batches are multi-channel, SSIM is used per channel
    and take the mean over channels.
    Args:
        batch1 (torch.Tensor): Tensor of shape [b, c, h, w]
        batch2 (torch.Tensor): Tensor of shape [b, c, h, w]
    Returns:
        anomaly_map (torch.Tensor): Tensor of shape [b, 1, h, w] of the anomaly map
    """
    dssim = ssim_module(batch1, batch2).mean(1, keepdim=True)
    ssim_map = 1 - 2 * dssim
    anomaly_map = 1 - ssim_map
    return anomaly_map


def metrics(config: Namespace, anomaly_maps: list = None, segmentations: list = None,
            anomaly_scores: list = None, labels: list = None) -> Union[None, float]:
    """
    Computes evaluation metrics, prints and logs the results.

    For pixel-level evaluation, both anomaly_maps and segmentations should be provided.
    For image-level evaluation, both anomaly_scores and labels should be provided.

    Args:
        anomaly_maps (list): list of anomaly map tensor batches of shape [b,c,h,w]
        segmentations (list): list of segmentation tensor batches of shape [b,c,h,w]
        anomaly_scores (list): list of anomaly score tensors of shape [b, 1]
        labels (list): list of label tensors of shape [b, 1]
    """

    print("\nEvaluation results: \n")

    # image-wise metrics
    if labels is not None:

        sample_ap = compute_average_precision(torch.cat(anomaly_scores), torch.cat(labels))
        print(f"sample-wise average precision: {sample_ap:.4f}")
        sample_auroc = compute_auroc(torch.cat(anomaly_scores), torch.cat(labels))
        print(f"sample-wise AUROC: {sample_auroc:.4f}\n")
        best_f1, _ = compute_best_dice(torch.cat(anomaly_scores), torch.cat(labels))
        print(f"Best F1-score for 100 thresholds: {best_f1:.4f}\n")

        log({'anom_val/sample_ap': sample_ap,
            'anom_val/sample-auroc': sample_auroc,
             'anom_val/best-F1': best_f1}, config)

    # pixel-wise metrics
    if segmentations is not None:

        if config.limited_metrics:
            pixel_ap = compute_average_precision(torch.cat(anomaly_maps), torch.cat(segmentations))
            print(f"pixel-wise average precision: {pixel_ap:.4f}\n")
            log({'anom_val/pixel-ap': pixel_ap}, config)

        else:
            pixel_ap = compute_average_precision(torch.cat(anomaly_maps), torch.cat(segmentations))
            print(f"pixel-wise average precision: {pixel_ap:.4f}")
            best_dice, threshold = compute_best_dice(torch.cat(anomaly_maps), torch.cat(segmentations))
            print(f"Best Dice score for 100 thresholds: {best_dice:.4f}")
            dice_5fpr = compute_dice_at_nfpr(torch.cat(anomaly_maps), torch.cat(segmentations))
            print(f"Dice score at 5% FPR: {dice_5fpr:.4f}")
            pixel_auroc = compute_auroc(torch.cat(anomaly_maps), torch.cat(segmentations))
            print(f"pixel-wise AUROC: {pixel_auroc:.4f}\n")

            log({'anom_val/pixel-ap': pixel_ap,
                'anom_val/pixel-auroc': pixel_auroc,
                 'anom_val/dice-5fpr': dice_5fpr,
                 'anom_val/best-dice': best_dice},
                config)

    if segmentations is not None and not config.limited_metrics:
        return threshold
    else:
        return None


def log(dict_to_log, config):
    """
    logs to wandb
    input is dict of values to log
    checks if value is Tensor Batch (len(shape)> 3) to treat it as image
    if value is not Tensor, it will treat it as number
    """
    for key, value in dict_to_log.items():
        if isinstance(value, Tensor):
            if len(value.shape) > 3:
                list_to_log = list(value[:config.num_images_log].float().cpu())
                config.logger.log({
                    key: [wandb.Image(img) for img in list_to_log]
                }, step=config.step)
        else:
            config.logger.log({
                key: value
            }, step=config.step)


def str_to_bool(value):
    # helper function to use boolean switches with arg_parse
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def seed_everything(seed: int) -> None:
    """Reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def denormalize(batch: Tensor, config: Namespace) -> Tensor:
    """
    Denormalizes a batch of tensor images that has been
    normalized with torchvision.utils.transofrm.Normalize.

    Args:
        batch (Tensor): batch of shape [B,C,W,H]
        config (Namespace): configuration object.

    Returns:
        denormed_batch (Tensor): the denormalized batch of shape [B,C,W,H]
    """

    if config.modality == 'CXR':
        sex = config.sex
        CXR_mean = {'male': 0.5228, 'female': 0.5312}
        CXR_std = {'male': 0.2872, 'female': 0.2851}
        denorm = T.Normalize((-1.0 * CXR_mean[sex] / CXR_std[sex]),
                             (1.0 / CXR_std[sex]))

    elif config.modality == 'OCT':
        OCT_mean = 0.1380
        OCT_std = 0.1440
        denorm = T.Normalize((-1.0 * OCT_mean / OCT_std), (1.0 / OCT_std))

    if config.modality == 'COL':

        HK_mean = np.array([0.5582, 0.3039, 0.2021])
        HK_std = np.array([0.3474, 0.2203, 0.1698])
        denorm = T.Normalize((-1.0 * HK_mean / HK_std), (1.0 / HK_std))

    if config.modality == 'MRI':
        if config.sequence == 't1':
            mean = 0.1419
            std = 0.3084

        else:
            mean = 0.1359
            std = 0.3018
        denorm = T.Normalize((-1.0 * mean / std), (1.0 / std))

    denormed_batch = denorm(batch)

    return denormed_batch


def memory():
    """
    Prints GPU memory information.
    """
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a
    print('Total Memory:', t / 10**9, 'Reserved Memory:', r / 10**9,
          'Allocated Memory:', a / 10**9, 'Free Memory inside Reserved', f / 10**9)
    return


def mean_std(dataloader: DataLoader, mask: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns mean and std of whole dataset.
    For grayscale, returns Tensor of size 3 with the same value on each chann
    Should be used before heavy augmentations, after resizing and cropping.

    Don't forget to disable drop_last for the dataloader for accurate results!

    source:
    kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html

    Args:
        dataloader (Dataloader): Dataloader instance.
        mask (bool, optional): True if dataloader provides masks or labels

    Returns:
        mean: Tensor with per channel mean
        std: Tensor with per channel standard deviation
    """

    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    if mask:
        for inputs, _ in dataloader:
            psum += inputs.sum(axis=[0, 2, 3])
            psum_sq += (inputs ** 2).sum(axis=[0, 2, 3])
    else:
        for inputs in dataloader:
            psum += inputs.sum(axis=[0, 2, 3])
            psum_sq += (inputs ** 2).sum(axis=[0, 2, 3])

    # extract img shape
    sample = next(iter(dataloader))
    rgb = False
    if mask:
        img_size = sample[0].shape[-1]
        if sample[0].shape[1] == 3:
            rgb = True
    else:
        img_size = sample.shape[-1]
        if sample.shape[1] == 3:
            rgb = True

    count = len(dataloader.dataset) * img_size * img_size

    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)
    if rgb:
        return total_mean, total_std
    else:
        return total_mean[0], total_std[0]


class GenericDataloader(DataLoader):
    """
    Generic Dataloader class to reduce boilerplate.
    Requires only Dataset object and configuration object for instantiation.

    Args:
        dataset (Dataset): dataset from which to load the data.
        config (Namespace): configuration object.
    """

    def __init__(self, dataset: Dataset, config: Namespace, shuffle: bool = True, drop_last: bool = False):
        super().__init__(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            pin_memory=False,
            num_workers=config.num_workers,
            drop_last=False)


def load_data(config):
    """
    Returns dataloaders according to splits. If config.normal_split == 1, train and validation
    dataloaders are the same and include the whole dataset. Similar for small and big testloaders
    if config.anomal_split == 1.

    Args:
        config (Namespace): configuration object.
    Returns: Tuple(train_loader, val_loader, big_testloader, small_testloader)

    """

    # conditional import for dataloaders according to modality
    if config.method == 'PII':
        if config.modality == 'MRI':
            from Dataloaders.PII_MRI import get_dataloaders
        elif config.modality == 'COLvid':
            from Dataloaders.PII_COL import get_dataloaders
            config.img_channels = 3
        elif config.modality == 'CXR':
            from Dataloaders.PII_CXR import get_dataloaders
        elif config.modality == 'RF':
            from Dataloaders.PII_RF import get_dataloaders
            config.img_channels = 3
    else:
        if config.modality == 'MRInoram':
            from Dataloaders.MRI_noram import get_dataloaders
        elif config.modality == 'MRI':
            from Dataloaders.MRI import get_dataloaders
        elif config.modality == 'OCT':
            from Dataloaders.OCT import get_dataloaders
        elif config.modality == 'COLvid':
            from Dataloaders.COLvid import get_dataloaders
            config.img_channels = 3
        elif config.modality == 'COL':
            from Dataloaders.COL import get_dataloaders
            config.img_channels = 3
        elif config.modality == 'CT':
            from Dataloaders.CT import get_dataloaders
        elif config.modality == 'CXR':
            from Dataloaders.CXR import get_dataloaders
        elif config.modality == 'RF':
            from Dataloaders.RF import get_dataloaders
            config.img_channels = 3

    # msg = "num_images_log should be lower or equal to batch size"
    # assert (config.batch_size >= config.num_images_log), msg

    print("Loading data...")
    t_load_data_start = time()

    if config.modality == 'CT':
        train_loader, val_loader, big_testloader, small_testloader = get_dataloaders(config)

        print('Training set: {} samples, Valid set: {} samples'.format(
            len(train_loader.dataset),
            len(val_loader.dataset),
        ))
        print('Big test-set: {} samples, Small test-set: set: {} samples.'.format(
            len(big_testloader.dataset),
            len(small_testloader.dataset),
        ))

        print(f'Loaded datasets in {time() - t_load_data_start:.2f}s')
        return train_loader, val_loader, big_testloader, small_testloader
    else:
        # For testloaders make batch_size equal to num_images_log, the number of images to log on wandb.
        # In evaluate.py there is the need to run through the dataloader once to calc threshold
        # when running through it again to produce and log images, it is more convenient to do it through
        # a single testloader iteration, instead of doing many testloader iterations until we aggregate the desired
        # number of images we want to log.
        temp = config.batch_size
        config.batch_size = config.num_images_log

        if config.anomal_split == 1.0:
            testloader = get_dataloaders(config, train=False)
            small_testloader = testloader
            big_testloader = testloader
            print(f'Test-set: {len(testloader.dataset)} samples.')
            msg = "batch_size too high, Testloader is empty."
            assert (len(testloader) != 0), msg

        else:

            big_testloader, small_testloader = get_dataloaders(config, train=False)

            print('Big test-set: {} samples, Small test-set: set: {} samples.'.format(
                len(big_testloader.dataset),
                len(small_testloader.dataset),
            ))

            msg = "anomal_split is too high or batch_size too high, Small testloader is empty."
            assert (len(small_testloader) != 0), msg

        if not config.eval or config.normal_fpr:
            # restore desired batch_size
            config.batch_size = temp

            if config.normal_split == 1.0:
                train_loader = get_dataloaders(config)
                print('Training set: {} samples'.format(
                    len(train_loader.dataset)
                ))
                train_loader = train_loader
                val_loader = train_loader
            else:
                train_loader, val_loader = get_dataloaders(config)
                print('Training set: {} samples, Valid set: {} samples'.format(
                    len(train_loader.dataset),
                    len(val_loader.dataset),
                ))

            print(f'Loaded datasets in {time() - t_load_data_start:.2f}s')
            return train_loader, val_loader, big_testloader, small_testloader

        else:

            print(f'Loaded datasets in {time() - t_load_data_start:.2f}s')
            return None, None, big_testloader, small_testloader


# def create_wandb_images(residual_tensor: Tensor,
#                         num_images_log: int, max_val: float) -> List[wandb.Image]:
#     """
#     Takes a tensor of residuals and returns a wandb.Image list.
#     It normalizes images using the first batch's max value so the
#     later batch's max values are not normalized to 255 by wandb's
#     internal PIL.Image.fromarray operation, which would result in all
#     logged images having the same range and appearing equally intensive.

#     Doing that, residuals of subsequent steps appear darker, given that
#     the training progresses as expected.

#     During the first image log, the max value of a batch of residuals
#     should be extracted and passed on every consequent call.

#     example:
#         $ if first validation step that the function is called:
#         $   max_intensity = anomaly_map.cpu().max()

#     Args:
#         residual_tensor (Tensor): residual tensor of shape [b, 1, w, h]
#         num_images_log (int): number of images to return
#         max_val (float): max intensity of first logged batch

#     Returns:
#         a list of wandb.Image objects with length num_images_log
#     """
#     if residual_tensor.is_cuda:
#         residual_tensor = residual_tensor.cpu()

#     image_array = residual_tensor[:num_images_log].squeeze(1)
#     image_array = image_array * 255 / max_val

#     for i in range(num_images_log):
#         image_array[i] = image_array[i].int() - image_array[i].min().int()
#     image_array = list(image_array.numpy().astype('uint8'))
#     images = [Image.fromarray(image) for image in image_array]

#     return [wandb.Image(image) for image in images]
