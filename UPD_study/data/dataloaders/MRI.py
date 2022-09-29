from typing import List, Tuple, Union
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from UPD_study.data.dataloaders.mri_preprocessing import (
    get_camcan_slices, get_brats_slices, get_atlas_slices)
from UPD_study.utilities.utils import GenericDataloader
from torch import Tensor
from argparse import Namespace


class NormalDataset(Dataset):
    """
    Dataset class for CamCAN Dataset.
    """

    def __init__(self, files: np.ndarray, config: Namespace):
        """
        Args:
            files(nd.array): array of shape [slices,1,H,W] already loaded
                             to ram with get_camcan_slices()
            config(Namespace): config object
        """

        self.files = files
        self.center = config.center

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Tensor:
        img = self.files[idx]
        img = torch.FloatTensor(img)
        # Center input
        if self.center:
            img = (img - 0.5) * 2

        return img


class AnomalDataset(Dataset):
    """
    Dataset class for the BraTS and ATLAS datasets.
    """

    def __init__(self, files: List, config: Namespace):
        """
        Args:
            files(List[np.ndarray, np.ndarray]): list of two arrays
            (slices and segmentations) of shapes [slices,1,H,W] loaded to ram

            config(Namespace): config object

        """
        self.images = files[0]
        self.segmentations = files[1]
        self.center = config.center

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        img = self.images[idx]
        img = torch.FloatTensor(img)
        # Center input
        if self.center:
            img = (img - 0.5) * 2

        seg = self.segmentations[idx]
        seg = torch.ByteTensor(seg)
        return img, seg


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

    if train:
        config.return_volumes = False
        # get array of slices

        if config.percentage != 100:
            config.return_volumes = True

        slices = get_camcan_slices(config)

        # percentage experiment: keep a specific percentage of the volumes, or a single volume.
        # for seed != 10 (stadard seed), take them from the back of the list
        if config.percentage != 100:
            if config.percentage == -1:  # single img scenario
                if config.seed == 10:
                    slices = slices[0]
                else:
                    slices = slices[-1]
                slices = [slices] * 100
                slices = np.concatenate(slices, axis=0)
            else:
                if config.seed == 10:
                    slices = slices[:int(len(slices) * (config.percentage / 100))]
                else:
                    slices = slices[-int(len(slices) * (config.percentage / 100)):]

                slices = np.concatenate(slices, axis=0)

        # keep slices with brain pixels in them
        slices = slices[np.sum(slices, axis=(1, 2, 3)) > 0]

        # calculate dataset split index
        split_idx = int(len(slices) * config.normal_split)

        if split_idx != len(slices):

            trainset = NormalDataset(slices[:split_idx], config)
            valset = NormalDataset(slices[split_idx:], config)

            train_dl = GenericDataloader(trainset, config)
            val_dl = GenericDataloader(valset, config)

            return train_dl, val_dl

        else:

            trainset = NormalDataset(slices, config)
            train_dl = GenericDataloader(trainset, config)

            return train_dl

    elif not train:

        # split before concating the volumes to keep complete patient samples in each subset
        config.return_volumes = True

        if config.sequence == 't1':
            if config.brats_t1:
                slices, segmentations = get_brats_slices(config)
            else:
                slices, segmentations = get_atlas_slices(config)
        elif config.sequence == 't2':
            slices, segmentations = get_brats_slices(config)

        split_idx = int(len(slices) * config.anomal_split)

        # if small part of anomal set is needed for validation (config.anomal_split != 1.0)
        if split_idx != len(slices):

            slices_big = np.concatenate(slices[:split_idx], axis=0)
            slices_small = np.concatenate(slices[split_idx:], axis=0)
            seg_big = np.concatenate(segmentations[:split_idx], axis=0)
            seg_small = np.concatenate(segmentations[split_idx:], axis=0)

            # keep slices with brain pixels in them
            non_zero_idx_s = np.sum(slices_small, axis=(1, 2, 3)) > 0
            slices_small = slices_small[non_zero_idx_s]
            seg_small = seg_small[non_zero_idx_s]

            non_zero_idx_b = np.sum(slices_big, axis=(1, 2, 3)) > 0
            slices_big = slices_big[non_zero_idx_b]
            seg_big = seg_big[non_zero_idx_b]

            for i in slices_big:
                if np.count_nonzero(i) < 5:
                    print(np.count_nonzero(i))
            big = AnomalDataset([slices_big, seg_big], config)
            small = AnomalDataset([slices_small, seg_small], config)

            big_test_dl = GenericDataloader(big, config, shuffle=config.shuffle)
            small_test_dl = GenericDataloader(small, config, shuffle=config.shuffle)

            del slices, segmentations, slices_small, seg_small

            return big_test_dl, small_test_dl

        else:
            return

            slices = np.concatenate(slices, axis=0)
            segmentations = np.concatenate(segmentations, axis=0)
            non_zero_idx = np.sum(slices, axis=(1, 2, 3)) > 0

            slices = slices[non_zero_idx]
            segmentations = segmentations[non_zero_idx]

            trainset = AnomalDataset([slices, segmentations], config)
            test_dl = GenericDataloader(trainset, config, shuffle=config.shuffle)

            return test_dl
