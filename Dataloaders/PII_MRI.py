
import sys
sys.path.append('/home/ioannis/Thesis/')

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


class PII_Dataset(Dataset):
    """
    Dataset class for the Healthy datasets.
    """

    def __init__(self, slices, config: Namespace):
        """
        Args:
            config(Namespace): config object

        config should include "sequence" and any MRI preprocessing options (eg. config.normalize)
        """

        # load all slices
        self.slices = slices  # [s,1,h,w]
        self.len = self.slices.shape[0]

        '''
        load each input and one random second image
        img1 : input images
        img2 : randomly sampled second images

        images need to be converted to ndarrays with shape CxHxW for the pii function input
        '''

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
        return self.len

    # def __getitem__(self, idx):

    def __getitem__(self, idx):
        # idx2 = np.random.randint(0, self.len)

        # img, mask = pii(self.slices[idx], self.slices[idx2], is_mri=True)
        img = self.slices[idx]
        #total_mask = np.zeros_like(img)
        idx2 = np.random.randint(0, self.len)
        img, mask = pii(img, self.slices[idx2], is_mri=True)
        #total_mask += mask

        # return torch.FloatTensor(img), torch.FloatTensor(mask)
        return torch.FloatTensor(img), torch.FloatTensor(mask)


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

        config should include "sequence"
        """

        if "is_gan" not in config or config.is_gan in [None, False]:
            self.is_gan = False
        else:
            self.is_gan = True

        self.images = files[0]
        self.segmentations = files[1]

        if config.sequence == 't1':
            mean = 0.1250
            std = 0.2481

        elif config.sequence == 't2':
            mean = 0.0785
            std = 0.1581

        self.transforms = T.Compose([
            T.Normalize(mean, std)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:

        img = self.images[idx]

        # if we are not training fanogan, stadardize imgs to N(0,1)
        if self.is_gan is False:
            img = self.transforms(torch.FloatTensor(img))
        else:
            img = torch.FloatTensor(img)

        seg = self.segmentations[idx]
        seg = torch.ByteTensor(seg)

        return img, seg


def get_dataloaders(config: Namespace,
                    train: bool = True) -> Union[DataLoader, Tuple[DataLoader,
                                                                   DataLoader]]:
    """
    Return pytorch Dataloader instances.

    """
    if train:
        # get array of slices
        slices = get_camcan_slices(config)

        # calculate dataset split index
        split_idx = int(len(slices) * config.normal_split)

        if split_idx != len(slices):

            trainset = PII_Dataset(slices[:split_idx], config)
            valset = PII_Dataset(slices[split_idx:], config)

            train_dl = GenericDataloader(trainset, config)
            val_dl = GenericDataloader(valset, config)

            return train_dl, val_dl

        else:

            trainset = PII_Dataset(slices, config)
            train_dl = GenericDataloader(trainset, config)

            return train_dl
    elif not train:

        # split before concating the volumes to keep complete patient samples in each subset
        config.return_volumes = True

        if config.sequence == 't1':
            slices, segmentations = get_atlas_slices(config)
        else:
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

            big = AnomalDataset([slices_big, seg_big], config)
            small = AnomalDataset([slices_small, seg_small], config)

            big_test_dl = GenericDataloader(big, config, shuffle=True)
            small_test_dl = GenericDataloader(small, config, shuffle=True)

            del slices, segmentations, slices_small, seg_small

            return big_test_dl, small_test_dl

        else:

            slices = np.concatenate(slices, axis=0)
            segmentations = np.concatenate(segmentations, axis=0)
            non_zero_idx = np.sum(slices, axis=(1, 2, 3)) > 0

            slices = slices[non_zero_idx]
            segmentations = segmentations[non_zero_idx]

            trainset = AnomalDataset([slices, segmentations], config)
            test_dl = GenericDataloader(trainset, config, shuffle=True)

            return test_dl
