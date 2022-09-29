from glob import glob
import os
from tifffile import imread
import numpy as np
from PIL import Image

x = ['train', 'valid', 'test']
for set in x:
    with open(f'DDR-dataset/DR_grading/{set}.txt', 'r') as labels:
        normals = [line.split(' ')[0] for line in labels if line[-2] == '0']
        for name in normals:
            current_path = f"DDR-dataset/DR_grading/{set}/{name}"
            target_path = f"DDR-dataset/healthy/{name}"
            os.rename(current_path, target_path)


segm_list = sorted(glob('DDR-dataset/lesion_segmentation/*/image/*'))
for seg in segm_list:
    x = seg.split('/')[-1].split('.')[0]
    segs = glob(f'DDR-dataset/lesion_segmentation/*/label/*/{x}.tif')
    total = imread(segs[0]) + imread(segs[1]) + imread(segs[2]) + imread(segs[3])
    total = np.where(total != 0, 255, 0)
    save_path = f'DDR-dataset/unhealthy/segmentations/{x}.png'
    Image.fromarray(total.save(save_path))
