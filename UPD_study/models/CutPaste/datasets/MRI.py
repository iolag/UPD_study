from typing import Tuple
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from UPD_study.data.dataloaders.mri_preprocessing import get_camcan_slices
from UPD_study.utilities.utils import GenericDataloader
from torch import Tensor
from argparse import Namespace
from cutpaste import CutPaste
from PIL import Image
from torchvision import transforms as T


class Cutpaste_Dataset(Dataset):
    """
    Dataset class for CamCAN Dataset.
    """

    def __init__(self, files: np.ndarray, config: Namespace):
        """
        Args:
            files(nd.array): array of MRI slices with shape [slices,1,H,W]
            config(Namespace): config object
        """

        self.files = files
        self.center = config.center
        self.cutpaste_transform = CutPaste(type=config.cutpaste_type)

        self.crop_size = (32, 32) if config.localization else (config.image_size, config.image_size)
        self.crop = T.RandomCrop(self.crop_size)
        self.transforms = T.Compose([
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Tensor:
        img = self.files[idx]
        # slices are returned by get_camcan_slices in 1xhxw numpy format and [0,1] range
        # need to repeat, permute and convert to PIL to apply cutpaste transformations
        img = np.tile(img, (3, 1, 1))
        img = img.transpose(1, 2, 0) * 255
        img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
        img_cropped = self.crop(img)
        cutpaste_list = self.cutpaste_transform(img_cropped)
        cutpaste_list = [self.transforms(i) for i in cutpaste_list]

        # Center input
        if self.center:
            cutpaste_list = [(i - 0.5) * 2 for i in cutpaste_list]
        return cutpaste_list


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

    config.return_volumes = False

    # get array of slices
    slices = get_camcan_slices(config)

    # keep slices with brain pixels in them
    slices = slices[np.sum(slices, axis=(1, 2, 3)) > 0]

    # calculate dataset split index
    split_idx = int(len(slices) * config.normal_split)

    cutpaste_trainset = Cutpaste_Dataset(slices[:split_idx], config)
    cutpaste_trainloader = GenericDataloader(cutpaste_trainset, config)

    cutpaste_valset = Cutpaste_Dataset(slices[split_idx:], config)
    cutpaste_valloader = GenericDataloader(cutpaste_valset, config)

    return cutpaste_trainloader, cutpaste_valloader
