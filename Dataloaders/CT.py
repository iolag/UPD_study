from typing import List, Tuple, Union
# import torch
from torch.utils.data import Dataset
# from torchvision import transforms as T
import numpy as np
from torch.utils.data import DataLoader
import sys
sys.path.append('/u/home/lagi/thesis')
from DatasetPreprocessing.mylungCT import get_munich_preproc_splits
from Utilities.utils import GenericDataloader
from torch import Tensor
from argparse import Namespace
import torchvision.transforms as T
import torch


class NormalDataset(Dataset):
    """
    Dataset class for Munich CT subset.
    """

    def __init__(self, normal_slices: torch.FloatTensor, config: Namespace):
        """
        Args:
            normal_slices (torch.FloatTensor): array of shape [slices,1,H,W] already loaded to ram
            config(Namespace): config object
        """
        self.normal_slices = normal_slices

        if "is_gan" not in config or config.is_gan in [None, False]:
            self.is_gan = False
        else:
            self.is_gan = True

        mean = 0.0077
        std = 0.0270

        self.stadardize = T.Normalize(mean, std)

    def __len__(self):
        return len(self.normal_slices)

    def __getitem__(self, idx) -> Tensor:

        image = self.normal_slices[idx]
        if not self.is_gan:
            image = self.stadardize(image)
        return image


class AnomalDataset(Dataset):

    def __init__(self,
                 anomal_slices_test,
                 anomal_masks_test,
                 normal_slices_test,
                 normal_masks_test, config: Namespace):
        """
        Args:
             Tensors of shape [slices,1,H,W]
            already loaded to ram
            config(Namespace): config object

        """

        if "is_gan" not in config or config.is_gan in [None, False]:
            self.is_gan = False
        else:
            self.is_gan = True

        self.images = torch.cat([anomal_slices_test, normal_slices_test])
        self.segmentations = torch.cat([anomal_masks_test, normal_masks_test])

        mean = 0.0077
        std = 0.0270

        self.stadardize = T.Normalize(mean, std)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:

        img = self.images[idx]
        seg = self.segmentations[idx]
        if not self.is_gan:
            img = self.stadardize(img)

        return img, seg.byte()


def get_dataloaders(config: Namespace,
                    train: bool = True) -> Union[DataLoader, Tuple[DataLoader,
                                                                   DataLoader]]:
    """
    Return pytorch Dataloader instances.

    Args:
        config (Namespace): Config object.
        train (bool): True for trainloaders, False for testloader with masks.
    Returns:
        train_dl (DataLoader) if train == True and config.spli_idx = 1
        train_dl, val_dl  (Tuple[DataLoader,DataLoader]) if train == True and
        config.spli_idx != 1
        test_dl (DataLoader) if train == False

    config must incl. "dataset_split", "num_workers", "batch_size", "sequence"
    """
    normal_slices, anomal_slices = get_munich_preproc_splits(config.image_size)

    normal_slices_train = normal_slices[1000:]
    anomal_slices_test = anomal_slices[0][:1000]
    anomal_masks_test = anomal_slices[1][:1000]
    normal_slices_test = normal_slices[:1000]
    normal_masks_test = torch.zeros_like(normal_slices_test)

    # calculate dataset split index
    split_idx = int(len(normal_slices_train) * config.normal_split)

    if split_idx != len(normal_slices_train):

        trainset = NormalDataset(normal_slices_train[:split_idx], config)
        valset = NormalDataset(normal_slices_train[split_idx:], config)

        train_dl = GenericDataloader(trainset, config)
        val_dl = GenericDataloader(valset, config)

    else:

        trainset = NormalDataset(normal_slices_train, config)
        train_dl = GenericDataloader(trainset, config)

    split_idx = int(len(anomal_slices_test) * config.anomal_split)

    # if small part of anomal set is needed for validation (config.anomal_split != 1.0)
    if config.anomal_split != 1.0:

        big = AnomalDataset(anomal_slices_test[:split_idx],
                            anomal_masks_test[:split_idx],
                            normal_slices_test[:split_idx],
                            normal_masks_test[:split_idx],
                            config)

        small = AnomalDataset(anomal_slices_test[split_idx:],
                              anomal_masks_test[split_idx:],
                              normal_slices_test[split_idx:],
                              normal_masks_test[split_idx:],
                              config)

        big_test_dl = GenericDataloader(big, config, shuffle=True)
        small_test_dl = GenericDataloader(small, config, shuffle=True)

    else:

        trainset = AnomalDataset(anomal_slices_test,
                                 anomal_masks_test,
                                 normal_slices_test,
                                 normal_masks_test,
                                 config)
        test_dl = GenericDataloader(trainset, config, shuffle=True)

    return train_dl, val_dl, big_test_dl, small_test_dl
