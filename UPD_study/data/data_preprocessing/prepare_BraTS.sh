# # Activate the virtual environment
# source /opt/anaconda3/etc/profile.d/conda.sh
# conda activate lagi

#Download BraTS 
kaggle datasets download -d awsaf49/brats20-dataset-training-validation
unzip brats20-dataset-training-validation.zip -d UPD_study/data/datasets/MRI
rm brats20-dataset-training-validation.zip
rm -r UPD_study/data/datasets/MRI/BraTS2020_ValidationData
mv UPD_study/data/datasets/MRI/BraTS2020_TrainingData UPD_study/data/datasets/MRI/BraTS
# fix a mistake in the naming
mv UPD_study/data/datasets/MRI/BraTS/MICCAI_BraTS2020_TrainingData/BraTS20_Training_355/W39_1998.09.19_Segm.nii UPD_study/data/datasets/MRI/BraTS/MICCAI_BraTS2020_TrainingData/BraTS20_Training_355/W39_1998.09.19_Segm.nii UPD_study/data/datasets/MRI/BraTS/MICCAI_BraTS2020_TrainingData/BraTS20_Training_355/W39_1998.09.19_Segm.nii UPD_study/data/datasets/MRI/BraTS/MICCAI_BraTS2020_TrainingData/BraTS20_Training_355/BraTS20_Training_355_seg.nii 

# Register and skullstrip
python UPD_study/data/data_preprocessing/prepare_data.py --dataset BraTS
