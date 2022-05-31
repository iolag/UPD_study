"""Create artificial anomalies via patch interpolation or poisson image interpolation.
Adapted from: https://github.com/jemtan/FPI/blob/master/self_sup_task.py
         and: https://github.com/jemtan/PII/blob/main/poissonBlend.py"""


# import os
# import sys
# this_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(this_dir))

from typing import Tuple

import numpy as np
import scipy
from scipy.sparse.linalg import spsolve


# TODO batch wise


def sample_location(img: np.ndarray, core_percent: float = 0.8, is_mri: bool = False) -> np.ndarray:
    """
    Sample a location for the core of the patch.
    The location is an array of shape (2,) where N is the number
    of patches.

    :param img: image tensor of shape (C, H, W)
    :param core_percent: percentage of the core region
    :param is_mri: whether the images are MRI slices, to sample core in foreground
    :return: array of shape (2,)
    """
    while True:
        dims = np.array(img.shape)
        core = core_percent * dims
        offset = (1 - core_percent) * dims / 2

        # Sample x-coordinates
        lo = int(np.floor(offset[-1]))
        hi = int(np.ceil(offset[-1] + core[-1]))
        cx = np.random.randint(lo, hi)

        # Sample y-coordinates
        lo = int(np.floor(offset[-2]))
        hi = int(np.ceil(offset[-2] + core[-2]))
        cy = np.random.randint(lo, hi)

        core = np.array([cx, cy])

        # if is_mri:
        #     if img[..., cx, cy] != 0.:
        #         break
        # else:
        #     break
        break
    return core


def sample_width(img: np.ndarray, min_p: float = 0.1, max_p: float = 0.4) -> float:
    """
    Sample a width for the patch.
    The width is a float between min_p and max_p of the image width

    :param img: image tensor of shape (H, W) or (C, H, W)
    :param min_p: minimum width percentage
    :param max_p: maximum width percentage
    :return: width
    """

    img_width = img.shape[-1]
    min_width = round(min_p * img_width)
    max_width = round(max_p * img_width)
    return np.random.randint(min_width, max_width)


# does not work with (H,W) due to last indexing operation
def create_patch_mask(img: np.ndarray, is_mri: bool = False) -> np.ndarray:
    """
    Create a mask for the given image.
    The mask is a tensor of shape (C, H, W) where C is the number of channels,
    H is the height of the image and W is the width of the image.
    The mask is a binary tensor with values 0 and 1.
    The mask is 1 if the patch is inside the image and 0 otherwise.

    :param img: image tensor of shape (C, H, W)
    :param patch_size: size of the patch
    :param is_mri: whether the images are MRI slices
    :return: mask tensor of shape (C, H, W)
    """
    # dims = img.shape

    # # Center of the patch
    # center = sample_location(img, is_mri=is_mri)

    # # Width of the patch
    # width = sample_width(img)

    # # Compute patch coordinates
    # coor_min = center - width // 2
    # coor_max = center + width // 2

    # # Clip coordinates to within image dims
    # coor_min = np.clip(coor_min, 0, dims[-2:])
    # coor_max = np.clip(coor_max, 0, dims[-2:])

    # # Create mask
    # mask = np.zeros(img.shape, dtype=np.float32)
    # mask[..., coor_min[0]:coor_max[0], coor_min[1]:coor_max[1]] = 1

    dims = img.shape

    # Center of the patch
    center = sample_location(img, is_mri=is_mri)  # (x,y)

    # Width of the patch
    width1 = sample_width(img)  # scalar
    width2 = width1

    # Compute patch coordinates
    coor_min1 = center - width1 // 2
    coor_max1 = center + width1 // 2
    coor_min2 = center - width2 // 2
    coor_max2 = center + width2 // 2

    # Clip coordinates to within image dims
    coor_min1 = np.clip(coor_min1, 0, dims[-2:])
    coor_max1 = np.clip(coor_max1, 0, dims[-2:])
    coor_min2 = np.clip(coor_min2, 0, dims[-2:])
    coor_max2 = np.clip(coor_max2, 0, dims[-2:])

    # Create mask
    mask = np.zeros(img.shape, dtype=np.float32)
    mask[..., coor_min1[0]:coor_max1[0], coor_min2[1]:coor_max2[1]] = 1

    return mask


def insert_laplacian_indexed(laplacian_op, mask: np.ndarray, central_val: float = 4):
    dims = np.shape(mask)  # (H, W) or (C, H, W)
    mask_flat = mask.flatten()
    inds = np.array((mask_flat > 0).nonzero())
    laplacian_op[..., inds, inds] = central_val
    laplacian_op[..., inds, inds + 1] = -1
    laplacian_op[..., inds, inds - 1] = -1
    laplacian_op[..., inds, inds + dims[-2]] = -1
    laplacian_op[..., inds, inds - dims[-2]] = -1
    laplacian_op = laplacian_op.tocsc()
    return laplacian_op


def pii(img1: np.ndarray, img2: np.ndarray, is_mri: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Performs poisson image interpolation between two images and
    returns the resulting images and the corresponding masks.

    :param img1: image tensor of shape  (C, H, W)
    :param img2: image tensor of shape  (C, H, W)
    :param is_mri: whether the images are MRI slices, to limit the procedure to the foreground
    :return: (img_pii, mask)
    """

    # Create mask
    patch_mask = create_patch_mask(img1, is_mri)  # shape  (C, H, W)
    interp = np.random.uniform(0.05, 0.95)
    mask = patch_mask * interp

    # Assure borders are 0 for poisson blending
    border_mask = np.zeros_like(mask)
    border_mask[..., 1:-1, 1:-1] = 1
    mask = border_mask * mask

    dims = np.shape(img1)
    clip_vals1 = (np.min(img1), np.max(img1))
    identity_matrix = scipy.sparse.identity(dims[-2] * dims[-1] * dims[-3]).tolil()

    # Flatten images
    img1_flat = img1.flatten()  # (H, W) -> (H * W)

    img2_flat = img2.flatten()  # (H, W) -> (H * W)
    mask_flat = mask.flatten()  # (H, W) -> (H * W)

    # Discrete approximation of gradient
    grad_matrix = insert_laplacian_indexed(identity_matrix, mask, central_val=0)
    grad_matrix.eliminate_zeros()  # Get rid of central, only identity or neighbours
    grad_mask = grad_matrix != 0  # (H * W, H * W)

    img1_grad = grad_matrix.multiply(img1_flat)  # Negative neighbour values
    # Add center value to sparse elements to get difference
    img1_grad = img1_grad + scipy.sparse.diags(img1_flat).dot(grad_mask)
    img2_grad = grad_matrix.multiply(img2_flat)
    img2_grad = img2_grad + scipy.sparse.diags(img2_flat).dot(grad_mask)

    # Mixing, favor the stronger gradient to improve blending
    alpha = np.max(mask_flat)
    img1_greater_mask = (1 - alpha) * np.abs(img1_grad) > alpha * np.abs(img2_grad)
    img1_guide = alpha * img2_grad - \
        img1_greater_mask.multiply(alpha * img2_grad) + \
        img1_greater_mask.multiply((1 - alpha) * img1_grad)

    img1_guide = np.squeeze(np.array(np.sum(img1_guide, 1)))
    img1_guide[mask_flat == 0] = img1_flat[mask_flat == 0]

    partial_laplacian = insert_laplacian_indexed(identity_matrix, mask,
                                                 central_val=4)

    x1 = spsolve(partial_laplacian, img1_guide)
    x1 = np.clip(x1, clip_vals1[0], clip_vals1[1])

    img_pii = np.reshape(x1, img1.shape)

    valid_label = (patch_mask * img1)[..., None] != (patch_mask * img2)[..., None]
    valid_label = np.any(valid_label, axis=-1)
    label = valid_label * mask

    x1 = spsolve(partial_laplacian, img1_guide)
    x1 = np.clip(x1, clip_vals1[0], clip_vals1[1])

    img_pii = np.reshape(x1, img1.shape)

    valid_label = (patch_mask * img1)[..., None] != (patch_mask * img2)[..., None]
    valid_label = np.any(valid_label, axis=-1)
    label = valid_label * mask

    img_pii = img_pii.astype(np.float32)

    # if MRI, clean the background
    if is_mri:
        img_pii[img1 == 0.] = 0.
        label[img1 == 0.] = 0.

    return img_pii, label
