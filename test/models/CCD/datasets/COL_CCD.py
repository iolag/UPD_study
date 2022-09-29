import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from augmentations import Rotation, Cutout1, Cutout, Gaussian_noise, CutPerm
import torchvision.transforms as transforms
import random


class CCD_Dataset(Dataset):
    def __init__(self, config, transform=None):
        self.transform = transform

        if config.cls_augmentation == 'noise':
            strong_aug = Gaussian_noise()
        elif config.cls_augmentation == 'cutperm':
            strong_aug = CutPerm()
        elif config.cls_augmentation == 'cutout':
            strong_aug = Cutout()
        elif config.cls_augmentation == 'rotation':
            strong_aug = Rotation()

        self.data = []

        self.imgs = np.asarray(get_files(config))

        # image will be resized later to config.image_size with RandomResizedCrop
        img_size = int(config.image_size // 0.875)

        self.initial_transform = T.Compose([
            T.Resize((img_size, img_size), T.InterpolationMode.LANCZOS),

            hide_blue_box(img_size),
            T.ToTensor(),
        ])

        for index in range(len(self.imgs)):
            img = Image.open(self.imgs[index])
            img = self.initial_transform(img)
            self.data.append(img)

        self.data = torch.stack(self.data)

        # num of strong augm. versions for the augm. classification task
        self.num_of_strong_aug_classes = 4

        self.augdata = []
        self.labels = []

        # create strong augment. image tensors for classif. task
        for i in range(self.data.shape[0]):

            tmp = self.data[i].unsqueeze(0)
            images = torch.cat([strong_aug(tmp, k)
                               for k in range(self.num_of_strong_aug_classes)])  # [4,c,h,w]
            # shift_labels = [0., 1., 2., 3.]
            shift_labels = np.asarray([1. * k for k in range(self.num_of_strong_aug_classes)])

            self.labels.append(shift_labels)
            self.augdata.append(images)

        self.data = torch.cat(self.augdata, axis=0)  # [samples*4,c,h,w]

        self.labels = np.concatenate(self.labels, axis=0)

    def __getitem__(self, idx):

        img = self.data[idx]
        label = self.labels[idx]
        label = torch.as_tensor(label).long()
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': label}
        return out

    def __len__(self):
        return len(self.data)


class AugmentedDataset(Dataset):
    """
    AugmentedDataset
    Returns an image together with an augmentation.
    """

    def __init__(self, dataset):
        super(AugmentedDataset, self).__init__()

        self.transform = dataset.transform
        # make it so in parent dataset it won't apply transform again during get_item:
        dataset.transform = None
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        sample = self.dataset.__getitem__(index)
        image = sample['image']
        sample['image'] = self.transform(image)  # α
        sample['image_augmented'] = self.transform(image)  # α'

        return sample


def get_train_dataloader(config):

    # SimCLR transforms
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=config.image_size, scale=[0.1, 1.0]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(0.2),
        # transforms.ToTensor(),
        # transforms.Normalize(**p['augmentation_kwargs']['normalize']),
        Cutout1(n_holes=1, length=75, random=True)
    ])

    dataset = CCD_Dataset(config, transform=transform)

    dataset = AugmentedDataset(dataset)

    return torch.utils.data.DataLoader(dataset,
                                       num_workers=8,
                                       batch_size=32,
                                       pin_memory=True,
                                       drop_last=True,
                                       shuffle=True)


class hide_blue_box(torch.nn.Module):

    """
    Crop the bluebox appearing in most Colonoscopy images.
    Return cropped img.

    The indexes to hide the blue boxes are a rough estimate
    after img inspection.
    """

    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size
        self.h_idx = 166 * image_size // 256
        self.w_idx = 90 * image_size // 256

    def forward(self, img):

        mask = np.ones((self.image_size, self.image_size, 3))
        # self.h_idx
        mask[self.h_idx:, 0: self.w_idx, :] = 0
        cropped_image = Image.fromarray(np.uint8(np.asarray(img) * mask))

        return cropped_image


def get_files(config, train: bool = True):

    pathfile = open(os.path.join(config.datasets_dir,
                                 'Colonoscopy',
                                 'labeled-images',
                                 'lower-gi-tract',
                                 'normal_image_paths.txt'))

    paths = pathfile.read().splitlines()

    # shuffle pathlist

    def myfunction():
        return 0.5
    random.shuffle(paths, myfunction)

    paths = paths[:2100]

    if train:
        return paths[500:]
