import os
from typing import List, Tuple, Union
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import numpy as np
from torch import Tensor
from argparse import Namespace
from UPD_study.utilities.utils import GenericDataloader
from glob import glob
import torch


def get_files(config: Namespace, train: bool = True) -> Union[List, Tuple[List, ...]]:
    """
    Return a list of all the paths of normal files.

    Args:
        config (Namespace): configuration object
        train (bool): True for train images, False for test images and labels
    Returns:
        images (List): List of paths of normal files.
        masks (List): (If train == True) List of paths of segmentations.

    """
    ap = "AP_" if config.AP_only else ""
    sup = "sup_" if config.sup_devices else "no_sup_"
    if config.sex == 'both':
        file_name = f'*_normal_train_{ap}{sup}'
        file = sorted(glob(os.path.join(config.datasets_dir,
                                        'CXR/normal_splits',
                                        file_name + '*.txt')))

    else:

        file_name = f'{config.sex}_normal_train_{ap}{sup}'

        file = sorted(glob(os.path.join(config.datasets_dir,
                                        'CXR/normal_splits',
                                        file_name + '*.txt')))

    paths1 = open(file[0]).read().splitlines()

    if config.sex == 'both':
        paths2 = open(file[1]).read().splitlines()

        # make sure even num of samples for both sexes

        if len(paths1) > len(paths2):
            paths1 = paths1[:len(paths2)]
        else:
            paths2 = paths2[:len(paths1)]

        for idx, path in enumerate(paths1):
            paths1[idx] = os.path.join(config.datasets_dir, 'CXR', path)
        for idx, path in enumerate(paths2):
            paths2[idx] = os.path.join(config.datasets_dir, 'CXR', path)

        if train:
            return paths1[200:] + paths2[200:]
    else:

        for idx, path in enumerate(paths1):
            paths1[idx] = os.path.join(config.datasets_dir, 'CXR', path)

        # returns here if we are taking normal files
        if train:
            return paths1[200:]

    # train == False --> test set

    if config.sex == 'both':

        file_name = f'*_anomal_{config.pathology}_{ap}{sup}'

        file = sorted(glob(os.path.join(config.datasets_dir,
                                        'CXR/anomal_splits',
                                        file_name + '*.txt')))
    else:

        file_name = f'{config.sex}_anomal_{config.pathology}_{ap}{sup}'

        file = sorted(glob(os.path.join(config.datasets_dir,
                                        'CXR/anomal_splits',
                                        file_name + '*.txt')))

    anom_paths1 = open(file[0]).read().splitlines()

    if config.sex == 'both':
        anom_paths2 = open(file[1]).read().splitlines()

        # make sure even num of samples for both sexes

        if len(anom_paths1) > len(anom_paths2):
            anom_paths1 = anom_paths1[:len(anom_paths2)]
        else:
            anom_paths2 = anom_paths2[:len(anom_paths1)]

        for idx, path in enumerate(anom_paths1):
            anom_paths1[idx] = os.path.join(config.datasets_dir, 'CXR', path)
        for idx, path in enumerate(anom_paths2):
            anom_paths2[idx] = os.path.join(config.datasets_dir, 'CXR', path)

        normal_paths = paths1[:200] + paths2[:200]
        anomal_paths = anom_paths1 + anom_paths2

        return normal_paths, anomal_paths[:400], [0] * len(normal_paths), [1] * 400

    else:
        for idx, path in enumerate(anom_paths1):

            anom_paths1[idx] = os.path.join(config.datasets_dir, 'CXR', path)

        return paths1[:200], anom_paths1[:200], [0] * 200, [1] * 200


class NormalDataset(Dataset):
    """
    Dataset class for the training set CXR images from CheXpert Dataset.
    """

    def __init__(self, files: List, config: Namespace):
        """
        Args
            files: list of image paths for healthy train CXR images
            config: Namespace() config object
        """

        self.center = config.center
        self.files = files

        self.transforms = T.Compose([
            T.Resize((config.image_size, config.image_size), T.InterpolationMode.LANCZOS),
            T.CenterCrop(config.image_size),
            T.ToTensor(),
        ])

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
        # Center input
        if self.center:

            image = (image - 0.5) * 2

        return image


class AnomalDataset(Dataset):
    """
    Dataset class for the test set CXR images from CheXpert Dataset.
    """

    def __init__(self,
                 normal_paths: List,
                 anomal_paths: List,
                 labels_normal: List,
                 labels_anomal: List,
                 config: Namespace):
        """
        Args:
            normal_paths (List): normal paths
            anomal_paths (List): anomal paths
            labels_normal (List): normal sample labels
            labels_anomal (List): anomal sample labels
            segmentations (List): binary segmentation masks
            config: Namespace() config object

        """

        self.center = config.center

        self.images = anomal_paths + normal_paths
        self.labels = labels_anomal + labels_normal

        self.image_transforms = T.Compose([
            T.Resize((config.image_size, config.image_size),
                     T.InterpolationMode.LANCZOS),
            T.CenterCrop(config.image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        """
        :param idx: Index of the file to load.
        :return: The loaded image and the binary segmentation mask.
        """

        image = Image.open(self.images[idx])
        image = self.image_transforms(image)

        if self.center:
            # Center input
            image = (image - 0.5) * 2

        # for compatibility with downstream methods that assume the existance
        # of ground truth segmentation, create fake image masks to provide as labels
        # negative = zero tensor, positive = Ident tensor
        if self.labels[idx] != 0:
            mask = torch.eye(image.shape[-1]).unsqueeze(0)
        else:
            mask = torch.zeros_like(image)

        return image, mask


def get_dataloaders(config: Namespace,
                    train: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Return pytorch Dataloader instances.

    Args:
        config (Namespace): Config object.
        train (bool): True for trainloaders, False for testloader with masks.
    Returns:
        train_dataloader, validation_dataloader  (Tuple[DataLoader,DataLoader]) if train == True
        big test_dataloader, small test_dataloader  (Tuple[DataLoader,DataLoader]) if train == False

    """

    if train:

        # get list of image paths
        trainfiles = get_files(config, train)

        # Deterministically shuffle before splitting  for the case of using both sexes
        torch.manual_seed(42)
        idx = torch.randperm(len(trainfiles))
        trainfiles = list(np.array(trainfiles)[idx])

        # percentage experiment
        # keep a specific percentage of the train files, or a single image.
        # for seed != 10 (stadard seed), index the list backwards
        if config.percentage != 100:
            if config.percentage == -1:  # single img scenario
                if config.seed == 10:
                    trainfiles = [trainfiles[0]] * 500
                else:
                    trainfiles = [trainfiles[-1]] * 500
                print(
                    f'Number of train samples ({len(trainfiles)}) lower than ',
                    f'batch size ({config.batch_size}). Repeating trainfiles 500 times.')
            else:
                if config.seed == 10:
                    trainfiles = trainfiles[:int(len(trainfiles) * (config.percentage / 100))]
                else:
                    trainfiles = trainfiles[-int(len(trainfiles) * (config.percentage / 100)):]
                if len(trainfiles) < config.batch_size:
                    print(
                        f'Number of train samples ({len(trainfiles)}) lower than ',
                        'batch size ({config.batch_size}). Repeating trainfiles 10 times.')
                    trainfiles = trainfiles * 10

        # calculate dataset split index
        split_idx = int(len(trainfiles) * config.normal_split)

        trainset = NormalDataset(trainfiles[:split_idx], config)
        valset = NormalDataset(trainfiles[split_idx:], config)

        train_dl = GenericDataloader(trainset, config)
        val_dl = GenericDataloader(valset, config)

        return train_dl, val_dl

    elif not train:
        # get list of img and mask paths
        normal, anomal, labels_normal, labels_anomal = get_files(config, train)

        # Deterministically shuffle before splitting for the case of using both sexes
        torch.manual_seed(42)
        idx = torch.randperm(len(normal))
        normal = list(np.array(normal)[idx])

        idx = torch.randperm(len(anomal))
        anomal = list(np.array(anomal)[idx])

        # calculate split indices
        split_idx = int(len(normal) * config.anomal_split)
        split_idx_anomal = int(len(anomal) * config.anomal_split)

        big = AnomalDataset(normal[:split_idx],
                            anomal[:split_idx_anomal],
                            labels_normal[:split_idx],
                            labels_anomal[:split_idx_anomal],
                            config)

        small = AnomalDataset(normal[split_idx:],
                              anomal[split_idx_anomal:],
                              labels_normal[split_idx:],
                              labels_anomal[split_idx_anomal:],
                              config)

        big_testloader = GenericDataloader(big, config, shuffle=False)
        small_testloader = GenericDataloader(small, config, shuffle=False)

        return big_testloader, small_testloader
