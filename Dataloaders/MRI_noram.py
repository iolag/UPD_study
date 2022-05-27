import os
from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torch import Tensor
from argparse import Namespace
import sys
from glob import glob
sys.path.append('/u/home/lagi/thesis')
from Utilities.utils import GenericDataloader


def get_files(config: Namespace, train: bool = True):

    if train:
        paths = glob(os.path.join(config.datasets_dir, 'cam_can_slices', '*.png'))

        return paths

    else:
        imgs = sorted(glob(os.path.join(config.datasets_dir, 'brats_slices', 'slices', '*.png')))
        masks = sorted(glob(os.path.join(config.datasets_dir, 'brats_slices', 'masks', '*.png')))
        return imgs, masks


class NormalDataset(Dataset):

    def __init__(self, files: List, config: Namespace):

        self.stadardize = config.stadardize

        self.files = files

        if config.sequence == 't1':
            mean = 0.1250
            std = 0.2481

        elif config.sequence == 't2':
            mean = 0.0785
            std = 0.1581

        self.initial_transform = T.Compose([
            T.Resize((config.image_size, config.image_size),
                     T.InterpolationMode.LANCZOS),
            T.CenterCrop(config.image_size),
            T.ToTensor(),
        ])

        self.norm = T.Normalize(mean, std)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Tensor:

        img = Image.open(self.files[idx])
        img = self.initial_transform(img)

        if self.stadardize:
            img = self.norm(img)

        return img


class AnomalDataset(Dataset):

    def __init__(self, images: List, masks: List, config: Namespace):

        self.stadardize = config.stadardize

        self.images = images
        self.masks = masks
        self.img_size = config.image_size

        if config.sequence == 't1':
            mean = 0.1250
            std = 0.2481

        elif config.sequence == 't2':
            mean = 0.0785
            std = 0.1581

        self.image_transforms = T.Compose([
            T.Resize((config.image_size, config.image_size),
                     T.InterpolationMode.LANCZOS),

            T.ToTensor(),

        ])
        self.norm = T.Normalize(mean, std)

        self.mask_transforms = T.Compose([
            T.Resize((config.image_size, config.image_size),
                     T.InterpolationMode.NEAREST),

            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        """
        :param idx: Index of the file to load.
        :return: The loaded img and the binary segmentation mask.
        """

        img = Image.open(self.images[idx])
        mask = Image.open(self.masks[idx])

        mask = self.mask_transforms(mask)
        img = self.image_transforms(img)

        # if we are not training fanogan, stadardize imgs to N(0,1)
        if self.stadardize:
            img = self.norm(img)

        return img, mask


def get_dataloaders(config: Namespace, train=True):

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
            train_dl = GenericDataloader(trainset, config, drop_last=True)

            return train_dl

    elif not train:
        # get list of img and mask paths
        images, masks = get_files(config, train)

        split_idx = int(len(images) * config.anomal_split)

        if split_idx != len(images):

            big = AnomalDataset(images[:split_idx],
                                masks[:split_idx], config)
            small = AnomalDataset(images[split_idx:],
                                  masks[split_idx:], config)

            big_testloader = GenericDataloader(big, config, shuffle=True)
            small_testloader = GenericDataloader(small, config, shuffle=True)

            return big_testloader, small_testloader
        else:
            testset = AnomalDataset(images, masks, config)

            test_dl = GenericDataloader(testset, config, shuffle=True)

            return test_dl
