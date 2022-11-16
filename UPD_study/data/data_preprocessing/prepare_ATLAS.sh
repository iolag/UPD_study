# # Activate the virtual environment
# source /opt/anaconda3/etc/profile.d/conda.sh
# conda activate lagi

#Download ATLAS
wget https://fcon_1000.projects.nitrc.org/indi/retro/ATLAS/releases/R2.0/ATLAS_R2.0_encrypted.tar.gz
openssl aes-256-cbc -md sha256 -d -a -in ATLAS_R2.0_encrypted.tar.gz -out ATLAS_R2.0.tar.gz
tar -xvzf ATLAS_R2.0.tar.gz -C UPD_study/data/datasets/MRI/ATLAS
rm ATLAS_R2.0.tar.gz
rm ATLAS_R2.0_encrypted.tar.gz
rm -r UPD_study/data/datasets/MRI/ATLAS/ATLAS_2/Testing

# Register and skullstrip
python UPD_study/data/data_preprocessing/prepare_data.py --dataset ATLAS
