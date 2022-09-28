
import torch
from torch.utils.data import Dataset
import numpy as np
from Models.PII.fpi_utils import pii
from argparse import Namespace
from torchvision import transforms as T
from torch.utils.data import DataLoader
from Utilities.utils import GenericDataloader
from DatasetPreprocessing.mri import (
    get_camcan_slices, get_brats_slices, get_atlas_slices)

from torch import Tensor
from typing import List, Tuple, Union


class NormalDataset(Dataset):
    """
    Dataset class for the Healthy datasets.
    """

    def __init__(self, files: np.ndarray, config: Namespace):
        """
        Args:
            config(Namespace): config object

        config should include "sequence" and any MRI preprocessing options (eg. config.normalize)
        """

        '''
        load each input and one random second image
        img1 : input images
        img2 : randomly sampled second images

        images need to be converted to ndarrays with shape CxHxW for the pii function input
        '''
        self.files = files
        self.stadardize = config.stadardize
        self.center = config.center

        if config.sequence == 't1':
            mean = 0.1250
            std = 0.2481

        elif config.sequence == 't2':
            mean = 0.0785
            std = 0.1581

        self.transforms = T.Compose([
            T.Normalize(mean, std)
        ])
        # img1 = []
        # img2 = []

        # for image1 in camcan_tensor:

        #     # load input image
        #     img1.append(image1)

        #     # load random second image corresponding to input
        #     img2_idx = np.random.randint(0, camcan_tensor.shape[0])
        #     image2 = camcan_tensor[img2_idx]
        #     img2.append(image2)

        # del camcan_tensor

        # apply pii using python multiprocessing
        # pool = mp.Pool(30)
        # results = pool.starmap(pii, [(img1[idx], img2[idx]) for idx in range(self.len)])
        # pool.close()
        # pool.join()

        # turn list of pii results to np.arrays and then to tensors
        # images = torch.from_numpy(np.array([item[0] for item in results]))
        # masks = torch.from_numpy(np.array([item[1] for item in results]))

        # # final Dataset files
        # self.images = images
        # self.masks = masks

    def __len__(self):
        return len(self.files)

    # def __getitem__(self, idx):

    def __getitem__(self, idx):
        # idx2 = np.random.randint(0, self.len)

        # img, mask = pii(self.slices[idx], self.slices[idx2], is_mri=True)
        img = self.files[idx]
        # total_mask = np.zeros_like(img)
        idx2 = np.random.randint(0, len(self))
        img, mask = pii(img, self.files[idx2], is_mri=True)
        # total_mask += mask

        mask = torch.FloatTensor(mask)
        if self.stadardize:
            img = self.transforms(torch.FloatTensor(img))
        else:
            img = torch.FloatTensor(img)
        if self.center:
            # Center input
            img = (img - 0.5) * 2

        return img, mask


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

        config should include "sequence" and "stadardize"
        """
        self.stadardize = config.stadardize
        self.images = files[0]
        self.segmentations = files[1]
        self.center = config.center
        self.atlas_exp = config.atlas_exp
        if config.sequence == 't1':
            mean = 0.1250
            std = 0.2481

        elif config.sequence == 't2':
            mean = 0.0785
            std = 0.1581

        elif config.sequence == 't1+t2':
            mean = 0.0785
            std = 0.1581

        self.transforms = T.Compose([
            T.Normalize(mean, std)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        img = self.images[idx]

        if self.stadardize:
            img = self.transforms(torch.FloatTensor(img))
        else:
            img = torch.FloatTensor(img)
        # Center input
        if self.center:

            img = (img - 0.5) * 2
            #atlas = (atlas - 0.5) * 2

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
        if config.sequence == 't1+t2':
            config.sequence = 't1'
            slices_t1 = get_camcan_slices(config)
            config.sequence = 't2'
            slices_t2 = get_camcan_slices(config)

            third_empty_channel = np.zeros_like(slices_t1)
            slices = np.concatenate([slices_t1, slices_t2, third_empty_channel], axis=1)
            zero_idx_t1 = np.sum(slices_t1[:, 0], axis=(1, 2)) > 0
            zero_idx_t2 = np.sum(slices_t2[:, 0], axis=(1, 2)) > 0
            slices = slices[zero_idx_t1 * zero_idx_t2]
            config.sequence = 't1+t2'
        else:
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

                else:
                    if config.seed == 10:
                        slices = slices[:int(len(slices) * (config.percentage / 100))]
                    else:
                        slices = slices[-int(len(slices) * (config.percentage / 100)):]

                    slices = np.concatenate(slices, axis=0)
            # if config.norm_vol:
            #     slices = np.concatenate([(volume - np.mean(volume)) / np.std(volume) for volume in slices])
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
        if config.get_images:
            config.slice_range = (72, 77)
        if config.sequence == 't1':
            if config.brats_t1:
                slices, segmentations = get_brats_slices(config)
            else:
                slices, segmentations = get_atlas_slices(config)
        elif config.sequence == 't2':
            slices, segmentations = get_brats_slices(config)
        elif config.sequence == 't1+t2':
            config.sequence = 't2'
            slices_t2, segmentations_t2 = get_brats_slices(config)
            config.sequence = 't1'
            slices_t1, segmentations_t1 = get_brats_slices(config)

            third_empty_channel = np.zeros_like(slices_t1)

            slices = np.concatenate([slices_t1, slices_t2, third_empty_channel], axis=2)
            segmentations = segmentations_t1
            config.sequence = 't1+t2'
        # if config.norm_vol:
        #     slices = np.concatenate([(volume - np.mean(volume)) / np.std(volume) for volume in slices])
        split_idx = int(len(slices) * config.anomal_split)

        # if small part of anomal set is needed for validation (config.anomal_split != 1.0)
        if split_idx != len(slices):

            slices_big = np.concatenate(slices[:split_idx], axis=0)
            slices_small = np.concatenate(slices[split_idx:], axis=0)
            seg_big = np.concatenate(segmentations[:split_idx], axis=0)
            seg_small = np.concatenate(segmentations[split_idx:], axis=0)

            # keep slices with brain pixels in them
            if config.sequence == 't1+t2':
                non_zero_idx_s_t1 = np.sum(slices_small[:, 0], axis=(1, 2)) > 0
                non_zero_idx_s_t2 = np.sum(slices_small[:, 1], axis=(1, 2)) > 0
                slices_small = slices_small[non_zero_idx_s_t1 * non_zero_idx_s_t2]
                seg_small = seg_small[non_zero_idx_s_t1 * non_zero_idx_s_t2]

                non_zero_idx_b_t1 = np.sum(slices_big[:, 0], axis=(1, 2)) > 0
                non_zero_idx_b_t2 = np.sum(slices_big[:, 1], axis=(1, 2)) > 0
                slices_big = slices_big[non_zero_idx_b_t1 * non_zero_idx_b_t2]
                seg_big = seg_big[non_zero_idx_b_t1 * non_zero_idx_b_t2]

            else:
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
