import os

from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import numpy as np
from torch import Tensor
from argparse import Namespace
from Utilities.utils import GenericDataloader
from glob import glob
import torch


def get_files(config: Namespace, train: bool = True) -> List or Tuple[List, List]:
    """
    Return a list of all the paths of normal files.

    Args:
        config (Namespace): configuration object
        train (bool): True for train images, False for test images and labels
    Returns:
        images (List): List of paths of normal files.
        masks (List): (If train == True) List of paths of segmentations.

    config should include "datasets_dir", "sex", "AP_only", "sup_devices"
    """
    if config.dataset == 'KAGGLE' or config.dataset == 'IDRID':
        pathfile = open(os.path.join(config.datasets_dir,
                                     'Retinal Fundus',
                                     'KAGGLE Fundus',
                                     'normal_train_samples_correct_ar.txt'))

        paths = pathfile.read().splitlines()

        for idx, path in enumerate(paths):
            paths[idx] = os.path.join(config.datasets_dir, path)

        if train and config.dataset == 'IDRID':
            return paths[1000:]
        elif train and config.dataset == 'KAGGLE':
            return paths[1000:]  # 1000 normal samples held for testset

        if config.dataset == 'KAGGLE':
            pathfile = open(os.path.join(config.datasets_dir,
                                         'Retinal Fundus',
                                         'KAGGLE Fundus',
                                         'abnormal_samples_correct_ar.txt'))

            anom_paths = pathfile.read().splitlines()

            for idx, path in enumerate(anom_paths):
                anom_paths[idx] = os.path.join(config.datasets_dir, path)

            return paths[:1000], anom_paths[:1000], [0] * 1000, [1] * 1000

        elif config.dataset == 'IDRID':
            anom_paths = sorted(glob(os.path.join(config.datasets_dir,
                                                  'Retinal Fundus',
                                                  'IDRID/A. Segmentation',
                                                  'imgs_center_cropped',
                                                  '*.png')))

            mask_paths = sorted(glob(os.path.join(config.datasets_dir,
                                                  'Retinal Fundus',
                                                  'IDRID/A. Segmentation',
                                                  'total_gt_cropped',
                                                  '*.png')))

            pathfile = open(os.path.join(config.datasets_dir,
                                         'Retinal Fundus',
                                         'IDRID',
                                         'normal_samples.txt'))
            normal_paths = pathfile.read().splitlines()

            for idx, path in enumerate(normal_paths):
                normal_paths[idx] = os.path.join(config.datasets_dir, path)

            return normal_paths[: len(anom_paths)], anom_paths, [0] * len(anom_paths), mask_paths

    # elif config.dataset == 'LAG':
    #     pathfile = open(os.path.join(config.datasets_dir,
    #                                  'Retinal Fundus',
    #                                  'KAGGLE Fundus',
    #                                  'normal_train_samples_correct_ar.txt'))

    #     paths = pathfile.read().splitlines()

    #     for idx, path in enumerate(paths):
    #         paths[idx] = os.path.join(config.datasets_dir, path)

    #     if train:
    #         return paths

    #     paths = glob(os.path.join(config.datasets_dir,
    #                               'Retinal Fundus',
    #                               'LAG',
    #                               'non_glaucoma/image',
    #                               '*.jpg'))

    #     anom_paths = glob(os.path.join(config.datasets_dir,
    #                                    'Retinal Fundus',
    #                                    'LAG',
    #                                    'suspicious_glaucoma/image',
    #                                    '*.jpg'))

    #     return paths[:1000], anom_paths[:1000], [0] * 1000, [1] * 1000

    elif config.dataset == 'LAG':

        paths = glob(os.path.join(config.datasets_dir,
                                  'Retinal Fundus',
                                  'LAG',
                                  'non_glaucoma/image',
                                  '*.jpg'))

        if train:
            return paths[300:]  # 300 normal samples held for testset

        anom_paths = glob(os.path.join(config.datasets_dir,
                                       'Retinal Fundus',
                                       'LAG',
                                       'suspicious_glaucoma/image',
                                       '*.jpg'))

        return paths[:300], anom_paths[:300], [0] * 300, [1] * 300


class NormalDataset(Dataset):
    """
    Dataset class for the Healthy CXR images
    from CheXpert Dataset.
    """

    def __init__(self, files: List, config: Namespace):
        """
        Args
            files: list of image paths for healthy colonoscopy images
            config: Namespace() config object

        config should include "image_size"
        """

        if "is_gan" not in config or config.is_gan in [None, False]:
            self.is_gan = False
        else:
            self.is_gan = True

        self.files = files

        self.transforms = T.Compose([
            T.Resize((config.image_size), T.InterpolationMode.LANCZOS),
            T.CenterCrop((config.image_size)),
            T.ToTensor(),
        ])
        if config.dataset in ['KAGGLE', 'IDRID']:
            mean = np.array([0.4662, 0.3328, 0.2552])
            std = np.array([0.2841, 0.2092, 0.1733])
        else:
            mean = np.array([0.5013, 0.3156, 0.2091])
            std = np.array([0.2052, 0.1535, 0.1185])

        self.stadardize = T.Normalize(mean, std)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Tensor:
        """
        Args:
            idx: Index of the file to load.
        Returns:
            image: image tensor of size []
        """

        image = Image.open(self.files[idx])
        image = self.transforms(image)

        if not self.is_gan:
            image = self.stadardize(image)

        return image


class AnomalDataset(Dataset):
    """
    Dataset class for the Segmented Colonoscopy
    images from the Hyper-Kvasir Dataset.
    """

    def __init__(self,
                 normal_paths: List,
                 anomal_paths: List,
                 labels_normal: List,
                 labels_anomal: List,
                 config: Namespace):
        """
        Args:
            images: list of image paths for segmented colonoscopy images
            masks: list of image paths for corresponding segmentations
            config: Namespace() config object

        config should include "image_size"
        """

        if "is_gan" not in config or config.is_gan in [None, False]:
            self.is_gan = False
        else:
            self.is_gan = True

        self.dataset = config.dataset
        self.images = normal_paths + anomal_paths
        self.labels = labels_normal + labels_anomal

        self.image_transforms = T.Compose([
            T.Resize((config.image_size), T.InterpolationMode.LANCZOS),
            T.CenterCrop((config.image_size)),
            T.ToTensor()
        ])

        self.mask_transforms = T.Compose([
            T.Resize((config.image_size), T.InterpolationMode.NEAREST),
            T.CenterCrop((config.image_size)),
            T.ToTensor()
        ])

        if config.dataset in ['KAGGLE', 'IDRID']:
            mean = np.array([0.4662, 0.3328, 0.2552])
            std = np.array([0.2841, 0.2092, 0.1733])
        else:
            mean = np.array([0.5013, 0.3156, 0.2091])
            std = np.array([0.2052, 0.1535, 0.1185])

        self.stadardize = T.Normalize(mean, std)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        """
        :param idx: Index of the file to load.
        :return: The loaded image and the binary segmentation mask.
        """

        image = Image.open(self.images[idx])
        image = self.image_transforms(image)

        # for compatibility, create image-like pixel masks to use as labels
        # should have 1 channel dim to be consistent with pixel AP calc

        if not self.is_gan:
            image = self.stadardize(image)

        if self.dataset == 'IDRID':
            if self.labels[idx] == 0:
                mask = torch.zeros_like(image)[0].unsqueeze(0)
            else:
                mask = Image.open(self.labels[idx])
                mask = self.mask_transforms(mask)
        else:
            if self.labels[idx] == 0:
                mask = torch.zeros_like(image)[0].unsqueeze(0)
            else:
                mask = torch.eye(image.shape[-1]).unsqueeze(0)

        return image, mask


def get_dataloaders(config: Namespace, train=True) -> DataLoader or Tuple[DataLoader, DataLoader]:
    """
    Return pytorch Dataloader instances.

    Args:
        config (Namespace): Config object.
        train (bool): True for trainloaders, False for testloader with masks.
    Returns:
        train_dl (DataLoader) if train == True and config.spli_idx = 1
        train_dl, val_dl  (Tuple[DataLoader,DataLoader]) if train == True and config.spli_idx != 1
        test_dl (DataLoader) if train == False

    config should include "dataset_split", "num_workers", "batch_size"
    """

    if train:

        # get list of image paths
        trainfiles = get_files(config, train)

        # calculate dataset split index
        split_idx = int(len(trainfiles) * config.normal_split)

        if split_idx != len(trainfiles):

            trainset = NormalDataset(trainfiles[:split_idx], config)
            valset = NormalDataset(trainfiles[split_idx:], config)

            train_dl = GenericDataloader(trainset, config)
            val_dl = GenericDataloader(valset, config)

            return train_dl, val_dl

        else:

            trainset = NormalDataset(trainfiles, config)
            train_dl = GenericDataloader(trainset, config)

            return train_dl

    elif not train:
        # get list of img and mask paths
        normal, anomal, labels_normal, labels_anomal = get_files(config, train)

        # calculate split indices
        split_idx = int(len(normal) * config.anomal_split)
        split_idx_anomal = int(len(anomal) * config.anomal_split)
        if split_idx != len(anomal):

            big = AnomalDataset(normal[:split_idx],
                                anomal[:split_idx_anomal],
                                labels_normal[:split_idx],
                                labels_anomal[:split_idx],
                                config)

            small = AnomalDataset(normal[split_idx:],
                                  anomal[split_idx_anomal:],
                                  labels_normal[split_idx:],
                                  labels_anomal[split_idx_anomal:],
                                  config)

            big_testloader = GenericDataloader(big, config, shuffle=True)
            small_testloader = GenericDataloader(small, config, shuffle=True)

            return big_testloader, small_testloader
        else:
            dataset = AnomalDataset(normal,
                                    anomal,
                                    labels_normal,
                                    labels_anomal,
                                    config)

            testloader = GenericDataloader(dataset, config, shuffle=True)

            return testloader
