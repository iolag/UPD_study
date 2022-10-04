# Activate the virtual environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lagi


# Register and skullstrip
python UPD_study/data/data_preprocessing/prepare_data.py --dataset BraTS
python UPD_study/data/data_preprocessing/prepare_data.py --dataset ATLAS
python UPD_study/data/data_preprocessing/prepare_data.py --dataset CamCAN
