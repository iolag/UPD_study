import multiprocessing
import numpy as np
import os
import subprocess
from typing import List, Optional
from time import time
from UPD_study import ROOT


def strip_skull_ROBEX(paths, out_dir: Optional[str] = None,
                      num_processes: Optional[int] = min(24, os.cpu_count())):
    """Use ROBEX to strip the skull of a T1 weighted brain MR Image. Takes ~100s

    DOES NOT WORK PROPERLY FOR T2 WEIGHTED IMAGES

    Args:
        paths (str or list of str)
        out_dir (str): output directory to save the results. If None, same as
                       input directory is used
        num_processes (int): Number of threads for multiprocessing
    """

    # Check if robex is installed
    if not os.path.exists(os.path.join(ROOT, 'data', 'data_preprocessing', 'ROBEX/runROBEX.sh')):
        raise RuntimeError(
            f"ROBEX not found at {os.path.join(ROOT, 'data', 'data_preprocessing', 'ROBEX/runROBEX.sh')},"
            " download and install it from https://www.nitrc.org/projects/robex "
        )

    if isinstance(paths, str):
        paths = [paths]

    # Split list into batches
    batches = [list(p) for p in np.array_split(paths, num_processes) if len(p) > 0]
    print(f"Skull stripping is using {len(batches)} cpu cores")

    processes = []
    # Start multiprocessing
    for i, batch in enumerate(batches):
        p = multiprocessing.Process(
            target=_strip_skull_ROBEX, args=(batch, i, out_dir,))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    print("Waiting for ROBEX to finish")
    for p in processes:
        p.join()


def _strip_skull_ROBEX(paths: List[str], i_process: int,
                       out_dir: Optional[str] = None):
    """Don't call this function yourself"""
    robex = os.path.join(ROOT, 'data', 'data_preprocessing', 'ROBEX/runROBEX.sh')

    # Start timer
    t_start = time()

    # Iterate over  all files
    for i, path in enumerate(paths):
        p_split = path.split('/')
        # Select directory to save the result
        d = ('/').join(p_split[:-1]) if out_dir is None else out_dir
        # Build the new file name
        f_name = p_split[-1].split('.nii')[0] + "_stripped.nii"
        if path.endswith('.gz'):
            f_name += '.gz'
        # Set together to build the path where the stripped file is saved
        save_path = os.path.join(d, f_name)
        if not os.path.exists(save_path):
            # Run ROBEX
            result = subprocess.run(
                [robex, path, save_path], capture_output=True, text=True
            )
            result.check_returncode()
        print(f"Process {i_process} finished {i + 1} of"
              f" {len(paths)} in {time() - t_start:.2f}s")
