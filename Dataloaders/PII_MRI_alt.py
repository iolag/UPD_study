
import sys
sys.path.append('/home/ioannis/Thesis/')

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np
from Models.PII.fpi_utilscp import pii
import multiprocessing as mp
from argparse import Namespace
from DatasetPreprocessing.mri import get_camcan_slices
from typing import Tuple, Union
from torch.utils.data import DataLoader
from Utilities.utils import GenericDataloader


class PII_Dataset(Dataset):
    """
    Dataset class for the Healthy datasets.
    """

    def __init__(self, slices, config: Namespace):
        """

        """

        # load all slices
        self.slices = slices  # [s,1,h,w]
        self.len = self.slices.shape[0]

    def __len__(self):

        return self.len

    # def __getitem__(self, idx):

    def __getitem__(self, idx):

        normal_img = self.slices[idx]
        changed_img = np.copy(normal_img)

        #num_patches = torch.randint(0, 3, (1,)).item()
        for i in range(1):
            idx2 = np.random.randint(0, self.len)
            changed_img, _ = pii(changed_img, self.slices[idx2], is_mri=True)

        return torch.FloatTensor(changed_img), torch.FloatTensor(normal_img)


def get_dataloaders(config: Namespace,
                    train: bool = True) -> Union[DataLoader, Tuple[DataLoader,
                                                                   DataLoader]]:
    """
    Return pytorch Dataloader instances.

    """
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
