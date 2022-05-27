
import sys
sys.path.append('/home/ioannis/Thesis/')
from glob import glob
import os
from typing import List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np
from torch.utils.data import DataLoader
from torch import Tensor
from argparse import Namespace
from Utilities.utils import GenericDataloader
from Models.PII.fpi_utils import pii
import random


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

    pathfile = open(os.path.join(config.datasets_dir,
                                 'Colonoscopy',
                                 'labeled-images',
                                 'lower-gi-tract',
                                 'normal_image_paths.csv'))

    paths = pathfile.read().splitlines()

    for idx, path in enumerate(paths):
        paths[idx] = os.path.join(config.datasets_dir, path)

    # add video frames
    paths = paths + glob(os.path.join(config.datasets_dir,
                                      'Colonoscopy/videos/normal_frames_cur/video*/*.png'))

    # shuffle pathlist
    def myfunction():
        return 0.5
    random.shuffle(paths, myfunction)

    if train:
        return paths[300:]

    else:

        abnormal_imgs = open(os.path.join(config.datasets_dir,
                                          'Colonoscopy',
                                          'segmented-images',
                                          'polyp_image_paths.csv')).read().splitlines()

        masks = open(os.path.join(config.datasets_dir,
                                  'Colonoscopy',
                                  'segmented-images',
                                  'polyp_mask_paths.csv')).read().splitlines()

        return abnormal_imgs[:300], paths[:300], masks[:300]


class PII_Dataset(Dataset):
    """
    Dataset class for the Healthy datasets.
    """

    def __init__(self, files: List, config: Namespace):
        """
        param files: list of img paths for healthy colonoscopy images
        param config: Namespace() config object

        config should include "image_size"
        """

        if "is_gan" not in config or config.is_gan in [None, False]:
            self.is_gan = False
        else:
            self.is_gan = True

        self.files = files

        HK_mean = np.array([0.5256, 0.2889, 0.1939])
        HK_std = np.array([0.3186, 0.2006, 0.1544])

        self.transforms = T.Compose([
            T.Resize((config.image_size, config.image_size),
                     T.InterpolationMode.LANCZOS),
            T.CenterCrop(config.image_size),
            hide_blue_box(config),
            T.ToTensor(),
        ])

        self.stadardize = T.Normalize(HK_mean, HK_std)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        img = Image.open(self.files[idx])
        img = self.transforms(img)

        idx2 = np.random.randint(0, len(self))

        img2 = Image.open(self.files[idx2])
        img2 = self.transforms(img2)

        img, mask = pii(img.numpy(), img2.numpy(), is_mri=False)

        return torch.FloatTensor(img), torch.FloatTensor(mask)


class AnomalDataset(Dataset):
    """
    Dataset class for the Segmented Colonoscopy
    images from the Hyper-Kvasir Dataset.
    """

    def __init__(self, images: List, normals: List, masks: List, config: Namespace):
        """
        param images: list of image paths for segmented colonoscopy images
        param masks: list of image paths for corresponding segmentations
        param config: Namespace() config object

        config should include "image_size"
        """

        if "is_gan" not in config or config.is_gan in [None, False]:
            self.is_gan = False
        else:
            self.is_gan = True

        self.images = images
        self.normals = normals
        self.norm_anom = images + normals
        self.masks = masks
        self.img_size = config.image_size

        # mean and std of normal samples
        HK_mean = np.array([0.4872, 0.2646, 0.1855])
        HK_std = np.array([0.3320, 0.2066, 0.1631])

        self.image_transforms = T.Compose([
            T.Resize((config.image_size, config.image_size),
                     T.InterpolationMode.LANCZOS),
            hide_blue_box(config),
            T.ToTensor(),
        ])

        self.stadardize = T.Normalize(HK_mean, HK_std)

        self.mask_transforms = T.Compose([
            T.Resize((config.image_size, config.image_size),
                     T.InterpolationMode.NEAREST),
            hide_blue_box(config),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.norm_anom)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        """
        :param idx: Index of the file to load.
        :return: The loaded image and the binary segmentation mask.
        """

        img = Image.open(self.norm_anom[idx])
        if idx > len(self.masks) - 1:
            mask = torch.zeros(size=(1, self.img_size, self.img_size)).int()
        else:
            mask = Image.open(self.masks[idx])
            mask = self.mask_transforms(mask)

        img = self.image_transforms(img)

        # if we are not training fanogan, stadardize imgs to N(0,1)
        if not self.is_gan:
            img = self.stadardize(img)

        return img, mask.int()[0].unsqueeze(0)  # make the mask shape [1,h,w]


def get_dataloaders(config: Namespace,
                    train=True) -> DataLoader or Tuple[DataLoader, DataLoader]:
    """
    Return pytorch Dataloader instances.

    config should include "normal_split", "anomal_split", "num_workers", "batch_size"

    Args:
        config (Namespace): Config object.
        train (bool): True for trainloaders, False for testloader with masks.
    Returns:
        train_dl (DataLoader) if train == True and config.spli_idx = 1
        train_dl, val_dl  (Tuple[DataLoader,DataLoader]) if train == True and
        config.spli_idx != 1
        test_dl (DataLoader) if train == False

    """

    if train:

        # get list of image paths
        trainfiles = get_files(config, train)

        # calculate dataset split index
        split_idx = int(len(trainfiles) * config.normal_split)

        if split_idx != len(trainfiles):

            trainset = PII_Dataset(trainfiles[:split_idx], config)
            valset = PII_Dataset(trainfiles[split_idx:], config)

            train_dl = GenericDataloader(trainset, config)
            val_dl = GenericDataloader(valset, config)

            return train_dl, val_dl

        else:

            trainset = PII_Dataset(trainfiles, config)
            train_dl = GenericDataloader(trainset, config)

            return train_dl

    elif not train:
        # get list of img and mask paths
        images, normals, masks = get_files(config, train)

        split_idx = int(len(images) * config.anomal_split)
        split_idx_norm = int(len(normals) * config.anomal_split)

        if split_idx != len(images):

            big = AnomalDataset(images[:split_idx],
                                normals[:split_idx_norm], masks[:split_idx], config)
            small = AnomalDataset(images[split_idx:],
                                  normals[split_idx_norm:], masks[split_idx:], config)

            big_testloader = GenericDataloader(big, config, shuffle=True)
            small_testloader = GenericDataloader(small, config, shuffle=True)

            return big_testloader, small_testloader
        else:
            testset = AnomalDataset(images, normals, masks, config)

            test_dl = GenericDataloader(testset, config, shuffle=True)

            return test_dl


class hide_blue_box(torch.nn.Module):

    """
    Crop the bluebox appearing in most Colonoscopy images.
    Return cropped image.

    The indexes to hide the blue boxes are a rough estimate
    after image inspection.
    """

    def __init__(self, config: Namespace):
        super().__init__()

        self.config = config
        self.h_idx = 166 * config.image_size // 256
        self.w_idx = 90 * config.image_size // 256

    def forward(self, image: Tensor) -> Tensor:

        mask = np.ones((self.config.image_size, self.config.image_size, 3))

        mask[self.h_idx:, 0: self.w_idx, :] = 0
        cropped_image = Image.fromarray(np.uint8(np.asarray(image) * mask))

        return cropped_image
