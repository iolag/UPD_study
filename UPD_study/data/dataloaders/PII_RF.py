import sys
sys.path.append('~/thesis/UAD_study/')
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
from UPD_study.utilities.utils import GenericDataloader
from UPD_study.models.PII.fpi_utils import pii
from multiprocessing import Pool, cpu_count
from functools import partial


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

    if config.dataset == 'DDR':
        norm_paths = sorted(
            glob(os.path.join(config.datasets_dir, 'DDR-dataset', 'healthy', '*.jpg')))
        anom_paths = sorted(glob(os.path.join(config.datasets_dir,
                            'DDR-dataset', 'unhealthy', 'images', '*.png')))

        segmentations = sorted(glob(os.path.join(config.datasets_dir, 'DDR-dataset',
                               'unhealthy', 'segmentations', '*.png')))
        if train:
            return norm_paths[757:]
        else:
            return norm_paths[:757], anom_paths, [0] * 757, [1] * 757, segmentations

    elif config.dataset == 'LAG':

        paths = sorted(glob(os.path.join(config.datasets_dir,
                                         'Retinal Fundus',
                                         'LAG',
                                         'non_glaucoma/image',
                                         '*.jpg')))

        if train:
            return paths[500:]  # 500 normal samples held for testset

        anom_paths = sorted(glob(os.path.join(config.datasets_dir,
                                              'Retinal Fundus',
                                              'LAG',
                                              'suspicious_glaucoma/image',
                                              '*.jpg')))
        attention_maps_anom = sorted(glob(os.path.join(config.datasets_dir,
                                                       'Retinal Fundus',
                                                       'LAG',
                                                       'suspicious_glaucoma/attention_map',
                                                       '*.jpg')))
        return paths[:500], anom_paths[:500], [0] * 500, [1] * 500, attention_maps_anom


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
        self.stadardize = config.stadardize

        self.files = files
        self.center = config.center
        self.preload_files = config.dataset == 'DDR'
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

        self.norm = T.Normalize(mean, std)

        if self.preload_files:

            with Pool(cpu_count()) as pool:
                self.preload = pool.map(partial(self.load_file), files)

    def load_file(self, file):

        image = Image.open(file)
        image = self.transforms(image)
        if self.stadardize:
            image = self.norm(image)

        if self.center:
            # Center input
            image = (image - 0.5) * 2
        return image

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.preload_files:
            img = self.preload[idx]
        else:
            img = Image.open(self.files[idx])
            img = self.transforms(img)

        idx2 = np.random.randint(0, len(self))

        if self.preload_files:
            img2 = self.preload[idx2]
        else:
            img2 = Image.open(self.files[idx2])
            img2 = self.transforms(img2)

        img, mask = pii(img.numpy(), img2.numpy(), is_mri=False)

        if self.center:
            # Center input
            img = (img - 0.5) * 2

        return torch.FloatTensor(img), torch.FloatTensor(mask)


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
                 segmentations: List,
                 config: Namespace):
        """
        Args:
            images: list of image paths for segmented colonoscopy images
            masks: list of image paths for corresponding segmentations
            config: Namespace() config object

        config should include "image_size"
        """

        self.stadardize = config.stadardize
        self.center = config.center
        self.segmentations = segmentations
        self.dataset = config.dataset
        self.images = anomal_paths + normal_paths
        self.labels = labels_anomal + labels_normal
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
        elif config.dataset == 'DDR':
            mean = np.array([0.3835, 0.2741, 0.1746])
            std = np.array([0.2831, 0.2082, 0.1572])
        else:
            mean = np.array([0.5013, 0.3156, 0.2091])
            std = np.array([0.2052, 0.1535, 0.1185])

        self.norm = T.Normalize(mean, std)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        """
        :param idx: Index of the file to load.
        :return: The loaded image and the binary segmentation mask.
        """

        image = Image.open(self.images[idx])
        # image = np.asarray(image)
        # for c in range(3):
        #     image[:, :, c] = self.clahe(image=image[:, :, c])['image']
        #     image[:, :, c] = Image.fromarray(image[:, :, c])
        #     #image[:,:,c]= self.clahe(image[:,:,c])
        # image = Image.fromarray(image)
        image = self.image_transforms(image)

        if self.stadardize:
            image = self.norm(image)

        if self.center:
            # Center input
            image = (image - 0.5) * 2

        # for compatibility, create image-like pixel masks to use as labels
        # should have 1 channel dim to be consistent with pixel AP calc

        # if self.dataset == 'IDRID':
        #     if self.labels[idx] == 0:
        #         mask = torch.zeros_like(image)[0].unsqueeze(0)
        #     else:
        #         mask = Image.open(self.labels[idx])
        #         mask = self.mask_transforms(mask)
        # else:
        if self.labels[idx] == 0:
            segmentation = torch.zeros_like(image)[0].unsqueeze(0)
        else:
            segmentation = Image.open(self.segmentations[idx])
            segmentation = self.mask_transforms(segmentation)

        return image, segmentation


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

        # percentage experiment: keep a specific percentage of the train files, or a single image.
        # for seed != 10 (stadard seed), take them from the back of the list
        if config.percentage != 100:
            if config.percentage == -1:  # single img scenario
                if config.seed == 10:
                    trainfiles = [trainfiles[0]]
                else:
                    trainfiles = [trainfiles[-1]]
                print(
                    f'Number of train samples ({len(trainfiles)}) lower than batch size ({config.batch_size}). Repeating trainfiles {config.batch_size} times.')
                trainfiles = trainfiles * config.batch_size
            else:
                if config.seed == 10:
                    trainfiles = trainfiles[:int(len(trainfiles) * (config.percentage / 100))]
                else:
                    trainfiles = trainfiles[-int(len(trainfiles) * (config.percentage / 100)):]
                if len(trainfiles) < config.batch_size:
                    print(
                        f'Number of train samples ({len(trainfiles)}) lower than batch size ({config.batch_size}). Repeating trainfiles 10 times.')
                    trainfiles = trainfiles * 10

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
        normal, anomal, labels_normal, labels_anomal, segmentations = get_files(config, train)

        # # Deterministically shuffle before splitting for the case of using both sexes
        # torch.manual_seed(42)
        # idx = torch.randperm(len(normal))
        # normal = list(np.array(normal)[idx])

        # idx = torch.randperm(len(anomal))
        # anomal = list(np.array(anomal)[idx])
        # segmentations = list(np.array(segmentations)[idx])

        # calculate split indices
        split_idx = int(len(normal) * config.anomal_split)
        split_idx_anomal = int(len(anomal) * config.anomal_split)

        if split_idx != len(anomal):

            big = AnomalDataset(normal[:split_idx],
                                anomal[:split_idx_anomal],
                                labels_normal[:split_idx],
                                labels_anomal[:split_idx_anomal],
                                segmentations[:split_idx_anomal],
                                config)

            small = AnomalDataset(normal[split_idx:],
                                  anomal[split_idx_anomal:],
                                  labels_normal[split_idx:],
                                  labels_anomal[split_idx_anomal:],
                                  segmentations[split_idx_anomal:],
                                  config)

            big_testloader = GenericDataloader(big, config, shuffle=config.shuffle)
            small_testloader = GenericDataloader(small, config, shuffle=config.shuffle)

            return big_testloader, small_testloader
        else:
            dataset = AnomalDataset(normal,
                                    anomal,
                                    labels_normal,
                                    labels_anomal,
                                    config)

            testloader = GenericDataloader(dataset, config, shuffle=False)

            return testloader
