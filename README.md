# Official Repository of: "Unsupervised Pathology Detection: A Deep Dive Into the State of the Art"

This repository contains code to reproduce experiments from the paper ["Unsupervised Pathology Detection: A Deep Dive Into the State of the Art"](https://ieeexplore.ieee.org/document/10197302) ([arXiv preprint](https://arxiv.org/abs/2303.00609)). 

In this work, we perform a comprehensive evaluation of the state of the art in unsupervised pathology detection. We find that recent feature-modeling methods achieve increased performance compared to previous work and are capable of benefiting from recently developed self-supervised pre-training algorithms, further increasing their performance.


### Citation
If you find our work useful for your research, please consider citing:
```
@ARTICLE{10197302,
  author={Lagogiannis, Ioannis and Meissen, Felix and Kaissis, Georgios and Rueckert, Daniel},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Unsupervised Pathology Detection: A Deep Dive Into the State of the Art}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2023.3298093}}
```
```
@misc{UPD_study,
  doi = {10.48550/ARXIV.2303.00609}, 
  url = {https://arxiv.org/abs/2303.00609},
  author = {Lagogiannis, Ioannis and Meissen, Felix and Kaissis, Georgios and Rueckert, Daniel}
  title = {Unsupervised Pathology Detection: A Deep Dive Into the State of the Art},
  publisher = {arXiv},
  year = {2023}
}
```

# Usage

Download this repository by running

```bash
git clone https://github.com/iolag/UPD_study/
```

in your terminal.

## Environment

Create and activate the Anaconda environment:

```bash
conda env create -f environment.yml
conda activate upd
```

Additionally, you need to install the repository as a package:

```bash
python3 -m pip install --editable .
```

To be able to use [Weights & Biases](https://wandb.ai) for logging follow the instructions at https://docs.wandb.ai/quickstart.
<!-- 
A quick guide on the folder and code structure can be found [here](structure.md). -->

## Data

### CheXpert 

~To download CheXpert you must first register at https://stanfordmlgroup.github.io/competitions/chexpert/. After you receive the subscription confirmation e-mail, download the downsampled version (11G) and extract the CheXpert-v1.0-small folder in data/datasets/CXR. No other steps are required and all splits are provided.~

#### Update 12/12/23

It seems that the small version of the dataset isn't available from the official source anymore. You can find it in [Kaggle](https://www.kaggle.com/datasets/ssttff/chexpertv10small).


### DDR 

To download and prepare the DDR dataset, run:

```bash
bash UPD_study/data/data_preprocessing/prepare_DDR.sh
```

### MRI: CamCAN, ATLAS, BraTS 

To download and preprocess ATLAS and BraTS, first download ROBEX from https://www.nitrc.org/projects/robex  and extract it in data/data_preprocessing/ROBEX. Then run:

```bash
bash UPD_study/data/data_preprocessing/prepare_ATLAS.sh
bash UPD_study/data/data_preprocessing/prepare_BraTS.sh
```
For ATLAS you need to apply for the data at https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html and receive the encryption password. During the run of prepare_ATLAS.sh you will be prompted to input the password.

For BraTS, Kaggle's API will be used to download the data. To be able to interact with the API, follow the instructions at https://www.kaggle.com/docs/api.

To download the CamCAN data, you need to apply for it at https://camcan-archive.mrc-cbu.cam.ac.uk/dataaccess/index.php. After you download them, put them in data/datasets/MRI/CamCAN and run:

```bash
python UPD_study/data/data_preprocessing/prepare_data.py --dataset CamCAN
```

## Experiments

To generate the "Main Results" from Tables 1 and 3 over all three seeds run:
```bash
bash UPD_study/experiments/main.sh 
```
Alternatively, for a single seed run:

```bash
bash UPD_study/experiments/main_seed10.sh 
```


To generate the "Self-Supervised Pre-training" results from Tables 2 and 4 over all three seeds run:
```bash
bash UPD_study/experiments/pretrained.sh
```
Alternatively, for a single seed run:

```bash
bash UPD_study/experiments/pretrained_seed10.sh      
```

To generate the "Complexity Analysis" results from Table 5 run:
```bash
bash UPD_study/experiments/benchmarks.sh
```

To generate "The Effects of Limited Training Data" results from Fig. 3 run:
```bash
bash UPD_study/experiments/percentage.sh
```
##

The repository contains PyTorch implementations for [VAE](https://arxiv.org/abs/1907.02796), [r-VAE](https://arxiv.org/abs/2005.00031), [f-AnoGAN](https://www.sciencedirect.com/science/article/abs/pii/S1361841518302640), [H-TAE-S](https://arxiv.org/abs/2207.02059), [FAE](https://arxiv.org/abs/2208.10992), [PaDiM](https://arxiv.org/abs/2011.08785), [CFLOW-AD](https://arxiv.org/abs/2107.12571), [RD](https://arxiv.org/abs/2201.10703), [ExpVAE](https://arxiv.org/abs/1911.07389), [AMCons](https://arxiv.org/abs/2203.01671), [PII](https://arxiv.org/abs/2107.02622), [DAE](https://openreview.net/forum?id=Bm8-t_ggzPD) and [CutPaste](https://arxiv.org/abs/2104.04015).

##
![A )](figures/repo_samples.png)

This repository is a work in progress and in need of more cleanup and documentation. If you face any issues or have any suggestions do not hesitate to contact me or open an issue.
