import multiprocessing
import os
from time import time
from typing import List, Optional

from dipy.align.imaffine import (
    AffineRegistration,
    MutualInformationMetric,
    transform_centers_of_mass,
)
from dipy.align.transforms import (
    AffineTransform3D,
    RigidTransform3D,
    TranslationTransform3D,
)
from dipy.viz import regtools
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


class MRIRegistrator:
    def __init__(
        self,
        template_path,
        brain_mask_path: Optional[str] = None,
        nbins: int = 32,
        sampling_proportion: Optional[int] = None,
        level_iters: List[int] = [100, 10, 5],
        sigmas: List[float] = [3.0, 1.0, 1.0],
        factors: List[int] = [4, 2, 1],
        verbose: bool = False
    ):
        """Class for a registrator to perform an affine registration based on
        mutual information with dipy.

        Args:
            template_path (str): path to the brain atlas file in NiFTI format
                                (.nii or .nii.gz)
            brain_mask_path (str): path to the brain mask for the template used
                                for skull stripping. Use None if no skull
                                stripping is needed for the template
            nbins (int): number of bins to be used to discretize the joint and
                        marginal probability distribution functions (PDF) to
                        calculate the mutual information.
            sampling_proportion (int): Value from 1 to 100. Percentage of voxels
                                    used to calculate the PDF. None is 100%
            level_iters (list of int): Number of optimization iterations at each
                                    resolution in the gaussian pyramid.
            sigmas (list of float): Standard deviations of the gaussian smoothing
                                    kernels in the pyramid.
            factors (list of int): Inverse rescaling factors for pyramid levels.
        """
        template_data = nib.load(template_path)
        self.template = template_data.get_fdata()
        self.template_affine = template_data.affine

        if self.template.ndim == 4:
            self.template = self.template.squeeze(-1)

        if brain_mask_path is not None:
            mask = nib.load(brain_mask_path).get_fdata()
            self.template = self.template * mask

        self.nbins = nbins
        self.sampling_proportion = sampling_proportion
        self.level_iters = level_iters
        self.sigmas = sigmas
        self.factors = factors

        self.verbose = verbose

    def _print(self, str):
        if self.verbose:
            print(str)

    @staticmethod
    def save_nii(f, img: np.ndarray, affine: np.ndarray, dtype: str):
        nib.save(nib.Nifti1Image(img.astype(dtype), affine), f)

    @staticmethod
    def load_nii(path: str, dtype: str = 'short'):
        # Load file
        data = nib.load(path, keep_file_open=False)
        volume = data.get_fdata(caching='unchanged',
                                dtype=np.float32).astype(np.dtype(dtype))
        affine = data.affine
        return volume, affine

    @staticmethod
    def overlay(template: np.ndarray, moving: np.ndarray, transformer=None):
        # Matplotlib params
        plt.rcParams['image.cmap'] = 'gray'
        plt.rcParams['image.interpolation'] = 'nearest'

        if transformer is not None:
            moving = transformer.transform(moving)
        regtools.overlay_slices(template, moving, None,
                                0, 'Template', 'Moving')
        regtools.overlay_slices(template, moving, None,
                                1, 'Template', 'Moving')
        regtools.overlay_slices(template, moving, None,
                                2, 'Template', 'Moving')
        plt.show()

    def transform(self, img, save_path, transformation, affine, dtype):
        """Transform a scan given a transformation and save it.

        Args:
            img (np.ndarray): Scan
            save_path (str): Path so save transformed scan
            transformation (AffineMap): Affine transformation map
            affine
            dtype (str): numpy datatype of transformed scan
        """
        # Save maybe
        if save_path is not None:
            img, _ = self.load_nii(img, dtype='float32')

        transformed = transformation.transform(img)

        # Find data type to save
        if (transformed - transformed.astype(np.short)).sum() == 0.:
            dtype = np.short
        else:
            dtype = np.dtype('<f4')

        self.save_nii(
            f=save_path,
            img=transformed,
            affine=affine,
            dtype=dtype
        )

    def register_batch(self, files: List[str],
                       num_cpus: int = min(12, os.cpu_count())):
        """Register a list of NiFTI files and save the registration result
        with a '_registered.nii' suffix"""

        with multiprocessing.Pool(processes=num_cpus) as pool:
            results = pool.starmap(
                self._register,
                zip(files, range(len(files)))
            )

        transformations = {}
        for r in results:
            transformations = {**transformations, **r}

        return transformations

    def _register(self, path: str, i_process: int):
        """Don't call yourself"""
        start = time()
        save_path = path.split('nii')[0][:-1] + '_registered.nii'
        if path.endswith('.gz'):
            save_path += '.gz'
        _, transformation = self(moving=path, save_path=save_path)
        print(f"Scan {i_process} done in {time() - start:.02f}s")

        return {save_path: transformation}

    def __call__(self, moving, moving_affine=None, save_path=None, show=False):
        """Register a scan

        Args:
            moving (np.array): 3D volume of a scan
            moving_affine (np.array): 4x4 affine transformation of volume2world
            show (bool): Plot the result
        """
        # Start timer
        t_start = time()

        # Maybe load moving image
        if isinstance(moving, str):
            moving, moving_affine = self.load_nii(moving, dtype="<f4")

        # First resample moving image to same resolution
        # identity = np.eye(4)
        # affine_map = AffineMap(identity,
        #                        self.template.shape, self.template_affine,
        #                        moving.shape, moving_affine)

        # Center of mass transform
        c_of_mass = transform_centers_of_mass(self.template, self.template_affine,
                                              moving, moving_affine)

        # Affine registration
        metric = MutualInformationMetric(self.nbins, self.sampling_proportion)
        affreg = AffineRegistration(metric=metric,
                                    level_iters=self.level_iters,
                                    sigmas=self.sigmas,
                                    factors=self.factors,
                                    verbosity=1 if self.verbose else 0)
        # 3D translational only transform
        self._print("3D translational transform")
        translation3d = TranslationTransform3D()
        translation = affreg.optimize(self.template, moving,
                                      translation3d, None,
                                      self.template_affine, moving_affine,
                                      starting_affine=c_of_mass.affine)

        # 3D rigid transform
        self._print("3D rigid transform")
        rigid3d = RigidTransform3D()
        rigid = affreg.optimize(self.template, moving, rigid3d, None,
                                self.template_affine, moving_affine,
                                starting_affine=translation.affine)

        # 3D affine transform
        self._print("3D affine transform")
        affine3d = AffineTransform3D()
        affine = affreg.optimize(self.template, moving, affine3d,
                                 None, self.template_affine,
                                 moving_affine,
                                 starting_affine=rigid.affine)

        registered = affine.transform(moving)
        transformation = affine

        self._print(f"Time for registration: {time() - t_start:.2f}s")

        if show:
            self.overlay(self.template, registered)
            plt.show()

        # Save maybe
        if save_path is not None:
            # Select the right datatype
            if np.abs(registered - registered.astype(np.short)).sum() == 0:
                dtype = 'short'
            else:
                dtype = "<f4"
            # Save
            self.save_nii(
                f=save_path,
                img=registered,
                affine=self.template_affine,
                dtype=dtype
            )

        return registered, transformation


# if __name__ == '__main__':
#     import os
#     DATAROOT = os.environ.get('DATAROOT')
#     atlas_path = os.path.join(
#         DATAROOT, "BrainAtlases/mni_icbm152_nlin_sym_09a/mni_icbm152_csf_tal_nlin_sym_09a.nii")
#     img_path = os.path.join(DATAROOT, "WMH/GE3T/100/orig/T1.nii.gz")
#     reg_path = os.path.join(DATAROOT, "reg.nii.gz")

#     reg = MRIRegistrator(
#         template_path=atlas_path,
#     )
#     reg(img_path, reg_path)
