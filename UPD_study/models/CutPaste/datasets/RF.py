import os
from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torch import Tensor
from argparse import Namespace
from UPD_study.utilities.utils import GenericDataloader
from glob import glob
from multiprocessing import Pool, cpu_count
from functools import partial
from cutpaste import CutPaste


def get_files(config: Namespace) -> List:
    """

    Return a list of paths of the normal samples used for training.

    Args:
        config (Namespace): configuration object

    Returns:
        images (List): List of paths of normal files.


    """

    norm_paths = sorted(
        glob(os.path.join(config.datasets_dir, 'RF', 'DDR-dataset', 'healthy', '*.jpg')))

    return norm_paths[757:]


class Cutpaste_Dataset(Dataset):
    """
    Dataset class for the cutpaste training set
    """

    def __init__(self, files: List, config: Namespace):
        """
        Args
            files: list of image paths for healthy RF images
            config: Namespace() config object
        """

        self.files = files
        self.center = config.center
        self.cutpaste_transform = CutPaste(type=config.cutpaste_type)
        self.crop_size = (32, 32) if config.localization else (config.image_size, config.image_size)

        self.transforms = T.Compose([
            T.Resize((config.image_size, config.image_size), T.InterpolationMode.LANCZOS),
            T.CenterCrop(config.image_size),
            T.RandomCrop(self.crop_size)
        ])

        self.to_tensor = T.ToTensor()

        with Pool(cpu_count()) as pool:
            self.preload = pool.map(partial(self.load_file), files)

    def load_file(self, file):

        image = Image.open(file)
        image = self.transforms(image)
        image = self.cutpaste_transform(image)
        image = [self.to_tensor(i) for i in image]
        # Center inputs
        if self.center:
            image = [(i - 0.5) * 2 for i in image]

        return image

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Tensor:
        """
        Args:
            idx: Index of the file to load.
        Returns:
            image: image tensor of size []
        """

        return self.preload[idx]


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

    # calculate dataset split index
    split_idx = int(len(trainfiles) * config.normal_split)

    cutpaste_trainset = Cutpaste_Dataset(trainfiles[:split_idx], config)
    cutpaste_trainloader = GenericDataloader(cutpaste_trainset, config)

    cutpaste_valset = Cutpaste_Dataset(trainfiles[split_idx:], config)
    cutpaste_valloader = GenericDataloader(cutpaste_valset, config)

    return cutpaste_trainloader, cutpaste_valloader
