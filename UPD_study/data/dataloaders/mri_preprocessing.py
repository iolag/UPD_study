import os
import sys
sys.path.append('~/thesis/UAD_study/')
from glob import glob
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Callable, List, Sequence, Tuple
import nibabel as nib
import numpy as np
from skimage.exposure import equalize_hist
from skimage.transform import resize


def get_camcan_files(cf) -> List[str]:

    path = os.path.join(sys.path[-1], cf.datasets_dir, 'MRI/CamCAN')
    files = glob(os.path.join(path, '*/',
                              f'*{cf.sequence.upper()}w_stripped_registered.nii.gz'))

    assert len(files) > 0, "No files found in CamCAN"
    return files


def get_brats_files(cf) -> Tuple[List[str], List[str]]:
    """Get all BRATS files in a given sequence (t1, t2, or flair).
    Args:
        path (str): Path to BRATS root directory
        sequence (str): One of "t1", "t2", or "flair"
    Returns:
        files (List[str]): List of files
        seg_files (List[str]): List of segmentation files
    """
    files = glob(os.path.join(sys.path[-1], cf.datasets_dir, 'MRI/BraTS/MICCAI_BraTS2020_TrainingData/*',
                              f'*{cf.sequence.lower()}*registered.nii.gz'))
    seg_files = [os.path.join(os.path.dirname(f), 'anomaly_segmentation.nii.gz') for f in files]
    assert len(files) > 0, "No files found in BraTS"
    return files, seg_files


def get_atlas_files(cf) -> List[str]:

    path = os.path.join(sys.path[-1], cf.datasets_dir, 'MRI/ATLAS/lesion')
    files = glob(os.path.join(path, '*/t01/',
                              f'*{cf.sequence.lower()}w_deface_stx_stripped_registered.nii.gz'))
    seg_files = [os.path.join(os.path.dirname(f), 'anomaly_segmentation.nii.gz') for f in files]
    assert len(files) > 0, "No files found in ATLAS"
    return files, seg_files


def load_nii(path: str, size: int = None, primary_axis: int = 0, dtype: str = "float32"):
    """Load a neuroimaging file with nibabel, [w, h, slices]
    https://nipy.org/nibabel/reference/nibabel.html
    Args:
        path (str): Path to nii file
        size (int): Optional. Output size for h and w. Only supports rectangles
        primary_axis (int): Primary axis (the one to slice along, usually 2)
        dtype (str): Numpy datatype
    Returns:
        volume (np.ndarray): Of shape [w, h, slices]
        affine (np.ndarray): Affine coordinates (rotation and translation),
                             shape [4, 4]
    """
    # Load file
    data = nib.load(path, keep_file_open=False)
    volume = data.get_fdata(caching='unchanged')  # [w, h, slices]
    affine = data.affine

    # Squeeze optional 4th dimension
    if volume.ndim == 4:
        volume = volume.squeeze(-1)

    # Resize if size is given and if necessary
    if size is not None and (volume.shape[0] != size or volume.shape[1] != size):
        volume = resize(volume, [size, size, size])

    # Convert
    volume = volume.astype(np.dtype(dtype))

    # Move primary axis to first dimension
    volume = np.moveaxis(volume, primary_axis, 0)

    return volume, affine


def load_nii_nn(path: str, size: int,
                is_atlas: bool = None,
                slice_range: Tuple[int, int] = None,
                normalize: bool = False,
                equalize_histogram: bool = False,
                dtype: str = "float32"):
    """
    Load a file for training. Slices should be first dimension, volumes are in
    MNI space and center cropped to the shorter side, then resized to size
    """
    vol = load_nii(path, primary_axis=2, dtype=dtype)[0]

    if slice_range is not None:
        vol = vol[slice_range[0]:slice_range[1]]

    vol = rectangularize(vol)

    # clear interpolation artifacts for atlas imgs

    if is_atlas is not None or is_atlas:
        vol[vol < 1e-1] = 0

    # clear general interpolation artifacts
    vol[vol < 1e-4] = 0

    if size is not None:
        vol = resize(vol, [vol.shape[0], size, size])

    # percentile normalization
    if normalize:
        vol = normalize_percentile(vol, 99)

    # Scale to 0,1
    if vol.max() > 0.:
        vol /= vol.max()

    # histogram equilization
    if equalize_histogram:
        vol = histogram_equalization(vol)

    return vol


def load_segmentation(path: str, size: int,
                      slice_range: Tuple[int, int] = None,
                      threshold: float = 0.4):
    """Load a segmentation file"""
    vol = load_nii_nn(path, size=size, slice_range=slice_range,
                      normalize=False, equalize_histogram=False)
    return np.where(vol > threshold, 1, 0)


def load_files_to_ram(files: Sequence, load_fn: Callable = load_nii_nn,
                      num_processes: int = cpu_count()) -> List[np.ndarray]:
    with Pool(num_processes) as pool:
        results = pool.map(load_fn, files)

    return results


def histogram_equalization(img):
    # Create equalization mask
    mask = np.where(img > 0, 1, 0)
    # Equalize
    img = equalize_hist(img, nbins=256, mask=mask)
    # Assure that background still is 0
    img *= mask

    return img


def normalize_percentile(img: np.ndarray, percentile: float = 99) -> np.ndarray:
    """Normalize an image or volume to a percentile foreground intensity.
    Args:
        img (np.ndarray): Image to normalize
        percentile (float): Percentile to normalize to
    Returns:
        img (np.ndarray): Normalized image
    """

    foreground = img[img > 0]
    maxi = np.percentile(foreground, percentile)

    img[img > maxi] = maxi

    img /= maxi

    return img


def rectangularize(img: np.ndarray) -> np.ndarray:
    """
    Center crop the image to the shorter side
    Args:
        img (np.ndarray): Image to crop, shape [slices, w, h]
    Returns:
        img (np.ndarray): Cropped image
    """
    # Get image shape
    w, h = img.shape[1:]

    if w < h:
        # Center crop height to width
        img = img[:, :, (h - w) // 2:(h + w) // 2]
    elif h < w:
        # Center crop width to height
        img = img[:, (w - h) // 2:(w + h) // 2, :]
    else:
        # No cropping
        pass

    return img


def get_camcan_slices(config):
    """Get all image slices of the CamCAN brain MRI dataset"""
    # Get all files
    files = get_camcan_files(config)
    hist = config.equalize_histogram if 'equalize_histogram' in config else False
    # Load all files
    volumes = load_files_to_ram(
        files,
        partial(load_nii_nn,
                size=config.image_size,
                slice_range=config.slice_range if 'slice_range' in config else None,
                normalize=config.normalize if 'normalize' in config else False,
                equalize_histogram=hist
                )
    )
    if "return_volumes" in config and config.return_volumes:
        return np.stack(volumes, axis=0)[:, :, None]
    else:
        images = np.concatenate(volumes, axis=0)[:, None]

        return images


def get_brats_slices(config):
    """Get all image slices and segmentations of the BraTS brain MRI dataset"""
    # Get all files
    files, seg_files = get_brats_files(config)
    hist = config.equalize_histogram if 'equalize_histogram' in config else False
    # Load all files
    volumes = load_files_to_ram(
        files,
        partial(load_nii_nn,
                size=config.image_size,
                slice_range=config.slice_range if 'slice_range' in config else None,
                normalize=config.normalize if 'normalize' in config else False,
                equalize_histogram=hist)
    )

    # Load all files
    seg_volumes = load_files_to_ram(
        seg_files,
        partial(load_segmentation,
                size=config.image_size,
                slice_range=config.slice_range if 'slice_range' in config else None)
    )

    if "return_volumes" in config and config.return_volumes:
        return np.stack(volumes, axis=0)[:, :, None], np.stack(seg_volumes, axis=0)[:, :, None]
    else:
        imgs = np.concatenate(volumes, axis=0)[:, None]
        segs = np.concatenate(seg_volumes, axis=0)[:, None]

        return imgs, segs


def get_atlas_slices(config):
    """Get all image slices and segmentations of the ATLAS brain MRI dataset"""
    # Get all files
    files, seg_files = get_atlas_files(config)
    hist = hist = config.equalize_histogram if 'equalize_histogram' in config else False
    # Load all files
    volumes = load_files_to_ram(
        files,
        partial(load_nii_nn,
                is_atlas=True,
                size=config.image_size,
                slice_range=config.slice_range if 'slice_range' in config else None,
                normalize=config.normalize if 'normalize' in config else False,
                equalize_histogram=hist)
    )

    # Load all files
    seg_volumes = load_files_to_ram(
        seg_files,
        partial(load_segmentation,
                size=config.image_size,
                slice_range=config.slice_range if 'slice_range' in config else None)
    )

    if "return_volumes" in config and config.return_volumes:
        imgs = np.stack(volumes, axis=0)[:, :, None]
        segs = np.stack(seg_volumes, axis=0)[:, :, None]

        return imgs, segs
    else:
        imgs = np.concatenate(volumes, axis=0)[:, None]
        segs = np.concatenate(seg_volumes, axis=0)[:, None]

        return imgs, segs


def get_samples(size: int = 128):
    f1, f2 = get_camcan_files(sequence="t1")[:2]
    load_fn = partial(load_nii_nn, size=size, equalize_histogram=True)
    img1 = load_fn(f1)[:, None]
    img2 = load_fn(f2)[:, None]
    return np.stack([img1, img2], axis=0)
