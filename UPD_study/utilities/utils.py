import os
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import torch
from time import time
from argparse import Namespace
from torch.utils.data import DataLoader, Dataset
from typing import Union, Tuple, Dict
from torch import Tensor
import random
import wandb
os.environ["WANDB_SILENT"] = "true"
import torchgeometry as tgm
from torch import nn
import torch
import pathlib
from UPD_study.utilities.metrics import (
    compute_average_precision,
    compute_auroc,
    compute_best_dice
)


def load_pretrained(model: nn.Module, config: Namespace) -> nn.Module:
    """
    Handles loading of self-supervised pre-trained weights for every method.

    Args:
        model: nn.Module instance of backbone to be loaded with CCD pre-trained weights
        config (Namespace): configuration object.
    Returns:
        nn.Module instance of backbone loaded with CCD pre-trained weights
    """
    # use this str to load pretrained backbone
    pretrained = f'CCD_{config.arch}_{config.modality}'
    if config.modality == 'MRI':
        pretrained += f'_{config.sequence}'
    pretrained += f'_{config.name_add}_seed:{config.seed}'

    # main directory path: '.../UPD_study/'
    main_dir_path = pathlib.Path(__file__).parents[1]

    if config.arch == 'vae':
        # load encoder
        enc_dict = model.encoder.state_dict()
        pretrained_dict = torch.load(os.path.join(main_dir_path,
                                                  'models/CCD/saved_models',
                                                  config.modality,
                                                  f'{pretrained}_encoder_.pth'))

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in enc_dict}
        for k, v in enc_dict.items():
            if k not in pretrained_dict:
                print(f'weights for {k} not in pretrained weight state_dict()')
        enc_dict.update(pretrained_dict)
        model.encoder.load_state_dict(enc_dict)

        # load bottleneck
        bn_dict = model.bottleneck.state_dict()
        pretrained_dict = torch.load(os.path.join(main_dir_path,
                                                  'models/CCD/saved_models',
                                                  config.modality,
                                                  f'{pretrained}_bottleneck_.pth'))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in bn_dict}
        bn_dict.update(pretrained_dict)
        model.bottleneck.load_state_dict(bn_dict)
        print('Pretrained backbone loaded.')

    else:
        # to properly pretrain fanogan with CCD, output dim needs to change
        if config.method == 'f-anoGAN':
            model.fc = nn.Linear(4 * 4 * 8 * config.dim, 1024).to(config.device)

        # load pretrained weight dict
        model_dict = model.state_dict()
        pretrained_dict = torch.load(os.path.join(main_dir_path,
                                                  'models/CCD/saved_models',
                                                  config.modality,
                                                  f'{pretrained}_.pth'))
        # check for missing weights
        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(f'weights for {k} not in pretrained weight state_dict()')

        # some workarounds for FAE to load layer0
        if config.method == 'FAE':
            layer_0_dict = model.layer0[0].state_dict()
            pretrained_dict_layer_0 = {'weight': v for k,
                                       v in pretrained_dict.items() if k == 'conv1.weight'}
            layer_0_dict.update(pretrained_dict_layer_0)
            model.layer0[0].load_state_dict(layer_0_dict)
            print('loaded weights for layer0.0.weight which is a conv layer, rest are BN related')

        # apply pre-trained weights to the model
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        # change fanogan fc layer back to default
        if config.method == 'f-anoGAN':
            model.fc = nn.Linear(4 * 4 * 8 * config.dim, config.latent_dim).to(config.device)

        print(f'Pre-trained backbone {pretrained} loaded.')
    return model


def save_model(model: Union[nn.Module, Dict[str, nn.Module]], config: Namespace) -> None:
    """
    Handles model saving for every method.

    Args:
        model: nn.Module instance or dict of nn.Module instances to be saved
        config (Namespace): configuration object.
    """
    if config.method == 'RD':
        torch.save(model['decoder'].state_dict(),
                   f'saved_models/{config.modality}/{config.name}_dec_.pth')
        torch.save(model['bn'].state_dict(),
                   f'saved_models/{config.modality}/{config.name}_bn_.pth')

    elif config.method == 'CCD' and config.backbone_arch == 'vae':
        torch.save(model['encoder'].state_dict(),
                   f'saved_models/{config.modality}/{config.name}_encoder_.pth')
        torch.save(model['bottleneck'].state_dict(),
                   f'saved_models/{config.modality}/{config.name}_bottleneck_.pth')
    elif config.method == 'AMCons':
        torch.save(model['encoder'].state_dict(),
                   f'saved_models/{config.modality}/{config.name}_enc_.pth')
        torch.save(model['decoder'].state_dict(),
                   f'saved_models/{config.modality}/{config.name}_dec_.pth')
    else:
        torch.save(model.state_dict(), f'saved_models/{config.modality}/{config.name}_.pth')


def load_model(config: Namespace) -> Tuple[nn.Module, ...]:
    """
    Handles model loading for every method.

    Args:
        config (Namespace): configuration object.
    Returns:
        nn.Module model instances for every method, loaded with saved weights
    """
    if config.method == 'RD':

        load_dec = torch.load(f'saved_models/{config.modality}/{config.name}_dec_.pth')
        load_bn = torch.load(f'saved_models/{config.modality}/{config.name}_bn_.pth')
        return load_dec, load_bn

    elif config.method == 'AMCons':

        load_enc = torch.load(f'saved_models/{config.modality}/{config.name}_enc_.pth')
        load_dec = torch.load(f'saved_models/{config.modality}/{config.name}_dec_.pth')
        return load_enc, load_dec

    elif config.method == 'f-anoGAN':

        if config.modality in ['CXR', 'RF']:
            generator_name = f'f-anoGAN_{config.modality}__seed:10_netG.pth'
            discriminator_name = f'f-anoGAN_{config.modality}__seed:10_netD.pth'
        elif config.modality == 'MRI':
            generator_name = f'f-anoGAN_{config.modality}_{config.sequence}__seed:10_netG.pth'
            discriminator_name = f'f-anoGAN_{config.modality}_{config.sequence}__seed:10_netD.pth'

        load_g = torch.load(os.path.join('saved_models',
                                         config.modality,
                                         generator_name))

        load_d = torch.load(os.path.join('saved_models',
                                         config.modality,
                                         discriminator_name))
        if config.eval:
            load_e = torch.load(f'saved_models/{config.modality}/{config.name}_netE.pth')
            return load_g, load_d, load_e

        # for the 2 run scenario where wgan is already trained and we want to train encoder
        else:
            return load_g, load_d

    else:
        return torch.load(f'saved_models/{config.modality}/{config.name}_.pth')


def misc_settings(config: Namespace) -> None:
    """
    Various settings set right after argument parsing, before loading datasets.
    Sets some general, experiment specific and modality specific settings
    creates name string as config.name to be used for saving,loading and
    the wandb logger as config.logger

    Args:
        config (Namespace): configuration object.
    """

    config.datasets_dir = os.path.join(os.path.expanduser('~/thesis/UAD_study/'), config.datasets_dir)

    # Select training device
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if config.speed_benchmark:
        config.modality = 'CXR'
        config.disable_wandb = True
        config.eval = True
        config.seed = 10

    if not config.eval:
        config.no_dice = True

    print(f"Using {config.device}.")

    # Multi purpose model name string
    if config.method == 'CCD':
        name = f'CCD_{config.backbone_arch}_{config.modality}'
    else:
        name = f'{config.method}_{config.modality}'

    if config.modality == 'MRI':
        name += f'_{config.sequence}'

    if config.load_pretrained:
        name += '_CCD'
    if config.modality == 'RF':
        config.img_channels = 3

    elif config.modality == 'RF' and config.method != 'f-anoGAN':
        if config.method != 'AMCons':
            config.center = True

    if config.percentage != 100:
        config.no_dice = True
        if config.percentage == -1:
            name += '_percentage:single_sample'
        else:
            name += f'_percentage:{config.percentage}'

    name += f'_{config.name_add}_seed:{config.seed}'

    # create saved_models folder
    os.makedirs(f'saved_models/{config.modality}', exist_ok=True)

    # init wandb logger
    wandb_name = name
    if config.method == 'VAE':
        if config.restoration:
            config.method = 'r-VAE'
            wandb_name = 'r-' + wandb_name
    if config.sequence == 't1':
        if config.brats_t1:
            wandb_name = wandb_name + '_BraTS'
        else:
            wandb_name = wandb_name + '_ATLAS'

    if not config.eval and not config.disable_wandb:
        logger = wandb.init(project=config.method, name=wandb_name, config=config, reinit=True)
    if config.eval and not config.disable_wandb:
        logger = wandb.init(project=config.method, name=f'{wandb_name}_eval', config=config, reinit=True)
    if config.disable_wandb:
        logger = wandb.init(mode="disabled")

    # keep name, logger, step in config to be used downstream
    config.name = name
    config.logger = logger
    config.step = 0
    return


# initialize SSIM Loss module
ssim_module = tgm.losses.SSIM(11, max_val=255)


def ssim_map(batch1: Tensor, batch2: Tensor) -> torch.Tensor:
    """
    Computes the anomaly map between two batches using SSIM.
    The torchgeometry.losses.SSIM module returns Structural Dissimilarity:

    DSSIM = (1 - SSIM)/2

    which is then turned to an anomaly map. If batches are multi-channel, SSIM is
    calculated per channel and then the mean over channels is returned.
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

    # disables pixel level evaluation for CXR
    if config.modality == 'CXR':
        segmentations = None

    # image-wise metrics
    if labels is not None:

        sample_ap = compute_average_precision(torch.cat(anomaly_scores), torch.cat(labels))
        print(f"sample-wise average precision: {sample_ap:.4f}")
        sample_auroc = compute_auroc(torch.cat(anomaly_scores), torch.cat(labels))
        print(f"sample-wise AUROC: {sample_auroc:.4f}\n")

        log({'anom_val/sample_ap': sample_ap,
            'anom_val/sample-auroc': sample_auroc}, config)

    # pixel-wise metrics
    if segmentations is not None:

        if config.no_dice:
            pixel_ap = compute_average_precision(torch.cat(anomaly_maps), torch.cat(segmentations))
            print(f"pixel-wise average precision: {pixel_ap:.4f}\n")
            log({'anom_val/pixel-ap': pixel_ap}, config)

        else:
            pixel_ap = compute_average_precision(torch.cat(anomaly_maps), torch.cat(segmentations))
            print(f"pixel-wise average precision: {pixel_ap:.4f}")
            best_dice, threshold = compute_best_dice(torch.cat(anomaly_maps), torch.cat(segmentations))
            print(f"Best Dice score for 100 thresholds: {best_dice:.4f}")

            log({'anom_val/pixel-ap': pixel_ap,
                 'anom_val/best-dice': best_dice},
                config)

    if segmentations is not None and not config.no_dice:
        return threshold
    else:
        return None


def log(dict_to_log: Dict[str, Union[float, Tensor]], config: Namespace) -> None:
    """
    Generic function that logs to wandb.
    Input is dict of values to log.
    Checks if value is Tensor Batch (len(shape)> 3) to treat it as image
    if value is not Tensor, it is treated as a scalar

    Args:
        dict_to_log (dict): keys are str with names of values to log, items are
                            either floats, eg. evaluation results or Tensor batches
                            of images to log
        config (Namespace): configuration object.

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
            from UPD_study.data.dataloaders.PII_MRI import get_dataloaders
        elif config.modality == 'CXR':
            from UPD_study.data.dataloaders.PII_CXR import get_dataloaders
        elif config.modality == 'RF':
            from UPD_study.data.dataloaders.PII_RF import get_dataloaders
    else:
        if config.modality == 'MRI':
            from UPD_study.data.dataloaders.MRI import get_dataloaders
        elif config.modality == 'CXR':
            from UPD_study.data.dataloaders.CXR import get_dataloaders
        elif config.modality == 'RF':
            from UPD_study.data.dataloaders.RF import get_dataloaders

    print("Loading data...")
    t_load_data_start = time()

    # For testloaders make batch_size equal to num_images_log, the number of images to log on wandb.
    # In evaluate.py there is the need to run through the dataloader once to calc threshold
    # when running through it again to produce and log images, it is more convenient to do it through
    # a single testloader iteration, instead of doing many testloader iterations until we aggregate the
    # desired number of images we want to log.

    temp = config.batch_size
    if config.method != 'DFR' or config.get_images:
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

    if not config.eval or config.method in ['DFR', 'CFLOW-AD']:
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
