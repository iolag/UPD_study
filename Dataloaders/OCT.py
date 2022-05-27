import os
from typing import List, Tuple
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch import Tensor
from argparse import Namespace
from Utilities.utils import GenericDataloader


def get_files(config: Namespace,
              train: bool = True) -> List or Tuple[List, List]:
    """
    Return a list of all the paths of normal files.

    Args:
        config (Namespace): configuration object
        train (bool): True for train images, False for test images and labels
    Returns:
        images (List): List of paths of normal files.
        masks (List): (If train == True) List of paths of segmentations.

    config should include "datasets_dir"
    """

    # TODO, img level sets, option to have all OCTS from dataset, not curated subset

    paths = open(os.path.join(config.datasets_dir,
                              'OCT/OCT2017/normal_paths.txt')).read().splitlines()

    # # whole dataset of 26565 samples
    paths = glob('/home/ioannis/Thesis/Datasets/OCT/OCT2017/*/NORMAL/*.jpeg')
    for idx, path in enumerate(paths):
        paths[idx] = os.path.join(config.datasets_dir, path)
    if train:
        return paths[500:]

    else:

        paths_anom = sorted(glob(os.path.join(config.datasets_dir,
                                              "OCT/DME/DME_testset/image/*.png")))

        masks = sorted(glob(os.path.join(config.datasets_dir,
                            "OCT/DME/DME_testset/binary_mask/*.png")))
        print(len(paths_anom))
        return paths[:500], paths_anom, masks


class NormalDataset(Dataset):
    """
    Dataset class for the Healthy OCT images
    from Kermany et al. Dataset.
    """

    def __init__(self, files: List, config: Namespace):
        """
        param files: list of img paths for healthy OCT images
        param config: Namespace() config object

        config should include "image_size"
        """
        self.stadardize = config.stadardize

        self.files = files

        # mean and std of normal samples
        OCT_mean = 0.1380
        OCT_std = 0.1440

        self.transforms = T.Compose([
            T.Resize((config.image_size, config.image_size),
                     T.InterpolationMode.LANCZOS),
            T.ToTensor(),

        ])

        self.norm = T.Normalize(OCT_mean, OCT_std)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Tensor:
        """
        :param idx: Index of the file to load.
        :return: The loaded img.
        """

        img = Image.open(self.files[idx])
        img = self.transforms(img)

        if self.stadardize:
            img = self.norm(img)

        return img


import torch


class AnomalDataset(Dataset):
    """
    Dataset class for the Segmented OCT images
    images from the Duke DME Dataset.
    """

    def __init__(self, normals: List, anomals: List, masks: List, config: Namespace):
        """
        param images: list of img paths for segmented OCT samples
        param masks: list of img paths for corresponding segmentations
        param config: Namespace() config object

        config should include "image_size"
        """
        self.stadardize = config.stadardize

        self.images = anomals
        self.normals = normals
        self.anom_norm = anomals + normals
        self.masks = masks

        # mean and std of normal samples
        OCT_mean = 0.1380
        OCT_std = 0.1440
        self.image_size = config.image_size
        self.image_transforms = T.Compose([
            T.Resize((config.image_size, config.image_size),
                     T.InterpolationMode.LANCZOS),
            T.CenterCrop(config.image_size),
            T.ToTensor(),

        ])

        self.norm = T.Normalize(OCT_mean, OCT_std)

        self.mask_transforms = T.Compose([

            T.Resize((config.image_size, config.image_size),
                     T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.images + self.normals)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        :param idx: Index of the file to load.
        :return: The loaded img and the binary segmentation mask.
        """

        if idx > len(self.masks) - 1:
            img = Image.open(self.anom_norm[idx])
            mask = torch.zeros(size=(1, self.image_size, self.image_size)).int()
        else:
            img = Image.open(self.images[idx])
            mask = Image.open(self.masks[idx])
            mask = self.mask_transforms(mask)

        img = self.image_transforms(img)

        if self.stadardize:
            img = self.norm(img)

        return img, mask


def get_dataloaders(config: Namespace,
                    train: bool = True) -> DataLoader or Tuple[DataLoader,
                                                               DataLoader]:
    """
    Return pytorch Dataloader instances.

    Args:
        config (Namespace): Config object.
        train (bool): True for trainloaders, False for testloader with masks.
    Returns:
        train_dl (DataLoader) if train == True and config.spli_idx = 1
        train_dl, val_dl  (Tuple[DataLoader,DataLoader]) if train == True
        and spli_idx != 1
        test_dl (DataLoader) if train == False

    config should include "dataset_split", "num_workers", "batch_size"
    """

    if train:

        # get list of img paths
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
        paths_norm, paths_anom, masks = get_files(config, train)

        split_idx = int(len(paths_norm) * config.anomal_split)
        split_idx_anomal = int(len(paths_anom) * config.anomal_split)

        if split_idx != len(paths_norm):

            big = AnomalDataset(paths_norm[:split_idx],
                                paths_anom[:split_idx_anomal], masks[:split_idx_anomal], config)
            small = AnomalDataset(paths_norm[split_idx:],
                                  paths_anom[split_idx_anomal:], masks[split_idx_anomal:], config)
            big_testloader = GenericDataloader(big, config, shuffle=False)
            small_testloader = GenericDataloader(small, config, shuffle=False)

            return big_testloader, small_testloader

        else:

            testset = AnomalDataset(paths_norm, paths_anom, masks, config)
            test_dl = GenericDataloader(testset, config, shuffle=False)

            return test_dl
