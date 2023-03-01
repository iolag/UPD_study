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
from cutpaste import CutPaste


def get_files(config: Namespace) -> Union[List, Tuple[List, ...]]:
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

        return paths1[200:] + paths2[200:]
    else:

        for idx, path in enumerate(paths1):
            paths1[idx] = os.path.join(config.datasets_dir, 'CXR', path)

        return paths1[200:]


class Cutpaste_Dataset(Dataset):
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
        self.cutpaste_transform = CutPaste(type=config.cutpaste_type)
        self.crop_size = (32, 32) if config.localization else (config.image_size, config.image_size)

        self.transforms = T.Compose([
            T.Resize((config.image_size, config.image_size), T.InterpolationMode.LANCZOS),
            T.CenterCrop(config.image_size),
            T.RandomCrop(self.crop_size)
        ])

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Tensor:
        """
        Args:
            idx: Index of the file to load.
        Returns:
            image: image tensor of size []
        """

        image = Image.open(self.files[idx]).convert('RGB')
        image = self.transforms(image)
        image = self.cutpaste_transform(image)
        image = [self.to_tensor(i) for i in image]

        # Center inputs
        if self.center:
            image = [(i - 0.5) * 2 for i in image]

        return image


def cutpaste_loader(config: Namespace) -> Tuple[DataLoader, DataLoader]:
    """
    Return pytorch Dataloader instances.

    Args:
        config (Namespace): Config object.
        train (bool): True for trainloaders, False for testloader with masks.
    Returns:
        train_dataloader, validation_dataloader  (Tuple[DataLoader,DataLoader]) if train == True
        big test_dataloader, small test_dataloader  (Tuple[DataLoader,DataLoader]) if train == False

    """

    # get list of image paths
    trainfiles = get_files(config)

    # Deterministically shuffle before splitting  for the case of using both sexes
    torch.manual_seed(42)
    idx = torch.randperm(len(trainfiles))
    trainfiles = list(np.array(trainfiles)[idx])

    # calculate dataset split index
    split_idx = int(len(trainfiles) * config.normal_split)

    cutpaste_trainset = Cutpaste_Dataset(trainfiles[:split_idx], config)
    cutpaste_trainloader = GenericDataloader(cutpaste_trainset, config)

    cutpaste_valset = Cutpaste_Dataset(trainfiles[split_idx:], config)
    cutpaste_valloader = GenericDataloader(cutpaste_valset, config)

    return cutpaste_trainloader, cutpaste_valloader
