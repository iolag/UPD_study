import argparse
from glob import glob
import os
import shutil

import nibabel as nib
import numpy as np
from tqdm import tqdm

from tifffile import imread
from PIL import Image

from UPD_study.data.data_preprocessing.registrators import MRIRegistrator
from UPD_study.data.data_preprocessing.robex import strip_skull_ROBEX
from UPD_study import ROOT


class BraTSHandler():
    def __init__(self, args):
        """
        https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation
        """

        if args.dataset_path is None:
            args.dataset_path = os.path.join(ROOT, 'data', 'datasets', 'MRI', 'BraTS')
        if not os.path.exists(os.path.join(args.dataset_path, 'MICCAI_BraTS2020_TrainingData')):
            raise RuntimeError("Run download_BraTS.sh to download and extract dataset or manually download"
                               f'  and extract to {args.dataset_path}')
        self.rename_lesions(args)
        self.registerBraTS(args)

    @staticmethod
    def rename_lesions(args):
        print("Renaming segmentation files in BraTS to "
              "'anomaly_segmentation_unregistered.nii'")
        lesion_files = sorted(glob(f"{args.dataset_path}/*/*/*_seg.nii"))
        target_files = [
            '/'.join(f.split('/')[:-1] + ['anomaly_segmentation_unregistered.nii']) for f in lesion_files]
        for lesion, target in zip(lesion_files, target_files):
            data = nib.load(lesion, keep_file_open=False)
            volume = data.get_fdata(caching='unchanged',
                                    dtype=np.float32).astype(np.dtype("short"))
            nib.save(nib.Nifti1Image(volume, data.affine), target)
            # shutil.copy(lesion, target)

    @staticmethod
    def registerBraTS(args):
        print("Registering BraTS")

        # Get all files
        files = sorted(glob(
            f"{args.dataset_path}/MICCAI_BraTS2020_TrainingData/*/*_t1.nii"))
        print(f"Found {len(files)} files.")
        if len(files) == 0:
            raise RuntimeError("0 files to be registered")

        # Initialize registrator

        template_path = os.path.join(
            ROOT, 'data', 'data_preprocessing', 'BrainAtlases/T1_brain.nii')
        registrator = MRIRegistrator(template_path=template_path)

        # Register files
        transformations = registrator.register_batch(files)

        for path, t in tqdm(transformations.items()):
            base = path[:path.rfind("t1")]
            folder = '/'.join(path.split('/')[:-1])
            # Transform T2 image
            path = base + "t2.nii"
            save_path = base + "t2_registered.nii"
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype='short'
            )
            # # Transform FLAIR image
            # path = base + "flair.nii"
            # save_path = base + "flair_registered.nii"
            # registrator.transform(
            #     img=path,
            #     save_path=save_path,
            #     transformation=t,
            #     affine=registrator.template_affine,
            #     dtype='short'
            # )
            # Transform segmentation
            path = os.path.join(
                folder, "anomaly_segmentation_unregistered.nii")
            save_path = os.path.join(
                folder, "anomaly_segmentation.nii")
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype='short'
            )


class CamCANHandler():
    def __init__(self, args):
        """https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/index.php"""
        if args.dataset_path is None:
            args.dataset_path = os.path.join(ROOT, 'data', 'datasets', 'MRI', 'CamCAN')

        print("Preparing CamCAN directory")
        self.prepare_CamCAN(args)
        print(f"Skull stripping for CamCAN {args.weighting} scans")
        # self.skull_strip_CamCAN(args)
        # self.register_CamCAN(args)

    @staticmethod
    def prepare_CamCAN(args):
        # Check if data is downloaded
        if not os.path.exists(os.path.join(args.dataset_path, 'cc700')):
            raise RuntimeError("Missing dataset. Apply for CamCAN data and "
                               f"place it into {args.dataset_path}")

        # Move the data to a 'normal' directory
        normal_dir = os.path.join(args.dataset_path, 'normal')
        os.makedirs(normal_dir, exist_ok=True)

        patient_dirs = glob(
            f"{os.path.join(args.dataset_path, 'cc700/mri/pipeline/release004/BIDS_20190411/anat')}/sub*/")
        for d in tqdm(patient_dirs):
            # Move all files from 'anat' dir to parent dir
            for f in glob(f"{d}anat/*"):
                shutil.move(f, d)

            # Remove the empty 'anat' directory
            shutil.rmtree(f"{d}anat/", ignore_errors=True)

            # Move the directory
            shutil.move(d, normal_dir)

    @staticmethod
    def register_CamCAN(args):

        print("Registering CamCAN")

        # Get all files
        files = glob(
            f"{os.path.join(args.dataset_path, 'normal')}/*/*T1w_stripped.nii.gz")
        print(f"Found {len(files)} files")

        if len(files) == 0:
            raise RuntimeError("Found 0 files")

        # Initialize the registrator
        template_path = os.path.join(
            ROOT, 'data', 'data_preprocessing', 'BrainAtlases/T1_brain.nii')
        registrator = MRIRegistrator(template_path=template_path)

        # Register files
        transformations = registrator.register_batch(files)

        for path, t in tqdm(transformations.items()):
            base = path[:path.rfind("T1")]
            # Transform T2 image
            path = base + "T2w_stripped.nii.gz"
            save_path = base + "T2w_stripped_registered.nii.gz"
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype='short'
            )

    def skull_strip_CamCAN(self, args):
        w = args.weighting
        if not isinstance(w, str):
            raise RuntimeError(f"Invalid value for --weighting {w}")
        # Get list of all files
        paths = glob(
            f"{os.path.join(args.dataset_path, 'normal')}/*/*{w.upper()}w.nii.gz")
        print(f"Found {len(paths)}")

        if len(paths) == 0:
            raise RuntimeError("No paths found")

        if w.lower() == 't1':
            # Run ROBEX
            strip_skull_ROBEX(paths)
        elif w.lower() == "t2":
            self.skull_strip_CamCAN_T2(paths)
        else:
            raise NotImplementedError("CamCAN skull stripping not implemented"
                                      f" for --weighting {w}")

    @staticmethod
    def skull_strip_CamCAN_T2(paths):
        """Skull strip the registered CamCAN T2 images with the results
        of the skull stripped registered CamCAN T1 images
        """
        for path in tqdm(paths):
            t1_stripped_path = f"{path[:path.rfind('T2w')]}T1w_stripped.nii.gz"
            t2_stripped_path = f"{path[:path.rfind('T2w')]}T2w_stripped.nii.gz"
            if not os.path.exists(t1_stripped_path):
                print(f"WARNING: No T1 skull stripped file found for {path}")
            # Load T2 weighted scan
            t2_data = nib.load(path)
            affine = t2_data.affine
            t2 = np.asarray(t2_data.dataobj, dtype=np.short)
            # Load T1 skull stripped scan
            t1_stripped = np.asarray(
                nib.load(t1_stripped_path).dataobj, dtype=np.short)
            t2_stripped = t2.copy()
            t2_stripped[t1_stripped == 0] = 0
            # Save skull stripped t2
            nib.save(nib.Nifti1Image(t2_stripped.astype(
                np.short), affine), t2_stripped_path)


class ATLASHandler():
    def __init__(self, args):
        """
        Obtain the password by completing the form at
        https://fcon_1000.projects.nitrc.org/indi/retro/atlas_download.html
        """
        if args.dataset_path is None:
            args.dataset_path = os.path.join(ROOT, 'data', 'datasets', 'MRI', 'ATLAS')

        if not os.path.exists(os.path.join(args.dataset_path, 'ATLAS_2')):
            raise RuntimeError("Run download_ATLAS.sh to download and extract dataset or manually download"
                               f' from https://fcon_1000.projects.nitrc.org/indi/retro/atlas_download.html'
                               f' and extract to {args.dataset_path}')
        self.rename_lesions(args)
        self.skull_strip_ATLAS(args)
        self.registerATLAS(args)

    @staticmethod
    def rename_lesions(args):
        print("Renaming segmentation files in ATLAS to "
              "'anomaly_segmentation_unregistered.nii'")
        lesion_files = sorted(glob(f"{args.dataset_path}/*/*/*/*/*/*/*lesion_mask.nii.gz"))
        target_files = [
            '/'.join(f.split('/')[:-1] + ['anomaly_segmentation_unregistered.nii.gz']) for f in lesion_files]
        for lesion, target in zip(lesion_files, target_files):
            data = nib.load(lesion, keep_file_open=False)
            volume = data.get_fdata(caching='unchanged',
                                    dtype=np.float32).astype(np.dtype("short"))
            nib.save(nib.Nifti1Image(volume, data.affine), target)
            # shutil.copy(lesion, target)

    @staticmethod
    def skull_strip_ATLAS(paths):

        # Get list of all files
        paths = sorted(glob(
            f"{os.path.join(args.dataset_path)}/*/*/*/*/*/*/*T1w.nii.gz"))
        print(f"Found {len(paths)}")

        if len(paths) == 0:
            raise RuntimeError("No paths found")

        # Run ROBEX
        strip_skull_ROBEX(paths)

    @staticmethod
    def registerATLAS(args):
        print("Registering skull-striped ATLAS")

        # Get all files
        files = sorted(glob(
            f"{args.dataset_path}/*/*/*/*/*/*/*T1w_stripped.nii.gz"))
        print(f"Found {len(files)} files.")
        if len(files) == 0:
            raise RuntimeError("0 files to be registered")

        # Initialize registrator
        template_path = os.path.join(
            ROOT, 'data', 'data_preprocessing', 'BrainAtlases/T1_brain.nii')
        registrator = MRIRegistrator(template_path=template_path)

        # Register files
        transformations = registrator.register_batch(files)

        for path, t in tqdm(transformations.items()):
            # base = path[:path.rfind("t1")]
            folder = '/'.join(path.split('/')[:-1])
            # Transform segmentation mask
            path = os.path.join(
                folder, "anomaly_segmentation_unregistered.nii.gz")
            save_path = os.path.join(
                folder, "anomaly_segmentation.nii.gz")
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype='short'
            )


class DDRHandler():
    def __init__(self, args):
        """
        Prepare the DDR Retinal Fundus dataset.
        http://www.sciencedirect.com/science/article/pii/S0020025519305377
        """
        if args.dataset_path is None:
            args.dataset_path = os.path.join(ROOT, 'data', 'datasets', 'RF')

        if not os.path.exists(os.path.join(args.dataset_path, 'DDR-dataset')):
            raise RuntimeError("Run download_DDR.sh to download and extract dataset or manually download"
                               f' from drive.google.com/drive/folders/1z6tSFmxW_aNayUqVxx6h6bY4kwGzUTEC'
                               f' and extract to {args.dataset_path}.')
        self.create_train_set(args)
        self.create_test_set(args)

    @staticmethod
    def create_train_set(args):
        # create train set folder
        os.makedirs(os.path.join(args.dataset_path, "DDR-dataset/healthy"), exist_ok=True)

        print('Creating train set...')
        subsets = ['train', 'valid', 'test']
        for set in subsets:
            with open(os.path.join(args.dataset_path, f'DDR-dataset/DR_grading/{set}.txt'), 'r') as labels:
                normals = [line.split(' ')[0] for line in labels if line[-2] == '0']
                for name in normals:
                    current_path = os.path.join(args.dataset_path, f"DDR-dataset/DR_grading/{set}/{name}")
                    target_path = os.path.join(args.dataset_path, f"DDR-dataset/healthy/{name}")
                    shutil.copyfile(current_path, target_path)

    @staticmethod
    def create_test_set(args):
        # create test set folders
        os.makedirs(os.path.join(args.dataset_path, "DDR-dataset/unhealthy/images"), exist_ok=True)
        os.makedirs(os.path.join(args.dataset_path, "DDR-dataset/unhealthy/segmentations"), exist_ok=True)

        print('Creating test set...')

        segm_list = sorted(glob(os.path.join(args.dataset_path, 'DDR-dataset/lesion_segmentation/*/image/*')))
        for seg in segm_list:
            img_name = seg.split('/')[-1].split('.')[0]
            target_path = os.path.join(args.dataset_path, f'DDR-dataset/unhealthy/images/{img_name}.png')
            shutil.copyfile(seg, target_path)
            segs = sorted(glob(os.path.join(args.dataset_path,
                          f'DDR-dataset/lesion_segmentation/*/label/*/{img_name}.tif')))
            if len(segs) != 4:
                raise RuntimeError(seg)
            total = imread(segs[0]) + imread(segs[1]) + imread(segs[2]) + imread(segs[3])
            total = np.where(total != 0, 255, 0)
            save_path = os.path.join(args.dataset_path, f'DDR-dataset/unhealthy/segmentations/{img_name}.png')
            Image.fromarray(total.astype(np.uint8)).save(save_path)


def prepare_data(args):
    if args.dataset == 'CamCAN':
        CamCANHandler(args)
    elif args.dataset == 'BraTS':
        BraTSHandler(args)
    elif args.dataset == 'ATLAS':
        ATLASHandler(args)
    elif args.dataset == 'DDR':
        DDRHandler(args)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['BraTS', 'CamCAN', 'ATLAS', 'DDR'])
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--weighting', type=str, default='t1',
                        choices=['t1', 't2', 'T1', 'T2', 'FLAIR'])

    args = parser.parse_args()

    prepare_data(args)
