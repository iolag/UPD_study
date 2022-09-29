"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
import random
from skimage.util import random_noise
import torch.nn as nn


class Cutout1(object):
    def __init__(self, n_holes, length, random=False):
        self.n_holes = n_holes
        self.length = length
        self.random = random

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        length = random.randint(1, self.length)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class Cutout(nn.Module):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, length):
        super(Cutout, self).__init__()
        self.n_holes = [0, 1, 2, 3]
        self.length = length

    def forward(self, img, aug_index=None):
        n_holes = self.n_holes[aug_index]
        output = self._cutout(img, n_holes)
        return output

    def _cutout(self, img, n_holes):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        img = torch.squeeze(img, dim=0)

        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(n_holes):
            y = np.random.randint(h - self.length)
            x = np.random.randint(w - self.length)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return torch.unsqueeze(img, dim=0)


class Gaussian_noise(nn.Module):
    def __init__(self):
        super(Gaussian_noise, self).__init__()

    def forward(self, img, aug_index=None):

        if aug_index is None:
            print('please provide aug_index')
        img = img.squeeze(dim=0)
        img = img.detach().cpu().numpy()
        img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize
        img = torch.from_numpy(img)

        # noise = self.noise_options[aug_index]
        if aug_index == 0:
            output = (img * 255).unsqueeze(dim=0)
        elif aug_index == 1:
            output = self._gaussian_noise(img)
        elif aug_index == 2:
            output = self._speckle_noise(img)
        elif aug_index == 3:
            output = self._salt_pepper_noise(img)
        return output.type(torch.float)

    def _gaussian_noise(self, img):

        gauss_img = torch.tensor(random_noise(img, mode='gaussian', mean=0, var=0.05, clip=True))
        # output = noise_aug(input)
        return (gauss_img * 255).unsqueeze(dim=0)

    def _speckle_noise(self, img):

        speckle_noise = torch.tensor(random_noise(img, mode='speckle', mean=0, var=0.05, clip=True))
        return (speckle_noise * 255).unsqueeze(dim=0)

    def _salt_pepper_noise(self, img):
        s_and_p = torch.tensor(random_noise(img, mode='s&p', salt_vs_pepper=0.5, clip=True))
        return (s_and_p * 255).unsqueeze(dim=0)


class Rotation(nn.Module):
    def __init__(self, max_range=4):
        super(Rotation, self).__init__()
        self.max_range = max_range
        self.prob = 0.5

    def forward(self, input, aug_index=None):

        _, _, H, W = input.size()

        if aug_index is None:
            aug_index = np.random.randint(4)

            output = torch.rot90(input, aug_index, (2, 3))

            _prob = input.new_full((input.size(0),), self.prob)
            _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)
            output = _mask * input + (1 - _mask) * output

        else:
            aug_index = aug_index % self.max_range
            output = torch.rot90(input, aug_index, (2, 3))

        return output


class CutPerm(nn.Module):
    def __init__(self, max_range=4):
        super(CutPerm, self).__init__()
        self.max_range = max_range
        self.prob = 0.5

    def forward(self, input, aug_index=None):

        _, _, H, W = input.size()

        if aug_index is None:
            aug_index = np.random.randint(4)

            output = self._cutperm(input, aug_index)

            _prob = input.new_full((input.size(0),), self.prob)
            _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)
            output = _mask * input + (1 - _mask) * output

        else:
            aug_index = aug_index % self.max_range
            output = self._cutperm(input, aug_index)

        return output

    def _cutperm(self, inputs, aug_index):

        _, _, H, W = inputs.size()
        h_mid = int(H / 2)
        w_mid = int(W / 2)

        jigsaw_h = aug_index // 2
        jigsaw_v = aug_index % 2

        if jigsaw_h == 1:
            inputs = torch.cat((inputs[:, :, h_mid:, :], inputs[:, :, 0:h_mid, :]), dim=2)
        if jigsaw_v == 1:
            inputs = torch.cat((inputs[:, :, :, w_mid:], inputs[:, :, :, 0:w_mid]), dim=3)

        return inputs
