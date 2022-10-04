# Activate the virtual environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lagi

# 


# BraTS T2
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mod MRI --percentage -1
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mod MRI --percentage 1
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mod MRI --percentage 10
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mod MRI --percentage 25
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mod MRI --percentage 50
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mod MRI --percentage 75
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mod MRI --percentage 100

python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mod MRI --percentage -1 
python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mod MRI --percentage 1 
python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mod MRI --percentage 10 
python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mod MRI --percentage 25
python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mod MRI --percentage 50 
python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mod MRI --percentage 75 
python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mod MRI --percentage 100 

python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mod MRI --percentage -1 
python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mod MRI --percentage 1 
python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mod MRI --percentage 10 
python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mod MRI --percentage 25
python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mod MRI --percentage 50 
python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mod MRI --percentage 75 
python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mod MRI --percentage 100 

python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mod MRI --percentage -1 
python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mod MRI --percentage 1 
python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mod MRI --percentage 10 
python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mod MRI --percentage 25
python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mod MRI --percentage 50 
python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mod MRI --percentage 75 
python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mod MRI --percentage 100 


# BraTS T1
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage -1
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage 1
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage 10
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage 25
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage 50
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage 75
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage 100

python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage -1 
python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage 1 
python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage 10 
python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage 25
python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage 50 
python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage 75 
python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage 100 

python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage -1 
python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage 1 
python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage 10 
python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage 25
python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage 50 
python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage 75 
python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage 100 

python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage -1 
python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage 1 
python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage 10 
python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage 25
python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage 50 
python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage 75 
python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mod MRI -seq t1 --brats_t1 f --percentage 100 

# CXR
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mod CXR --percentage -1
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mod CXR --percentage 1
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mod CXR --percentage 10
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mod CXR --percentage 25
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mod CXR --percentage 50
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mod CXR --percentage 75
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mod CXR --percentage 100

python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mod CXR --percentage -1 
python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mod CXR --percentage 1 
python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mod CXR --percentage 10 
python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mod CXR --percentage 25
python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mod CXR --percentage 50 
python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mod CXR --percentage 75 
python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mod CXR --percentage 100 

python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mod CXR --percentage -1 
python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mod CXR --percentage 1 
python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mod CXR --percentage 10 
python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mod CXR --percentage 25
python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mod CXR --percentage 50 
python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mod CXR --percentage 75 
python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mod CXR --percentage 100 

python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mod CXR --percentage -1 
python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mod CXR --percentage 1 
python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mod CXR --percentage 10 
python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mod CXR --percentage 25
python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mod CXR --percentage 50 
python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mod CXR --percentage 75 
python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mod CXR --percentage 100 

# RF
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mor RF --percentage -1
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mor RF --percentage 1
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mor RF --percentage 10
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mor RF --percentage 25
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mor RF --percentage 50
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mor RF --percentage 75
python UPD_study/models/DAE/DAEtrainer.py --shuffle t -ev t -mor RF --percentage 100

python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mor RF --percentage -1 
python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mor RF --percentage 1 
python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mor RF --percentage 10 
python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mor RF --percentage 25
python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mor RF --percentage 50 
python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mor RF --percentage 75 
python UPD_study/models/RD/RDtrainer.py --shuffle t -ev t -mor RF --percentage 100 

python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mor RF --percentage -1 
python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mor RF --percentage 1 
python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mor RF --percentage 10 
python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mor RF --percentage 25
python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mor RF --percentage 50 
python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mor RF --percentage 75 
python UPD_study/models/FAE/FAEtrainer.py --shuffle t -ev t -mor RF --percentage 100 

python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mor RF --percentage -1 
python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mor RF --percentage 1 
python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mor RF --percentage 10 
python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mor RF --percentage 25
python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mor RF --percentage 50 
python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mor RF --percentage 75 
python UPD_study/models/VAE/VAEtrainer.py --shuffle t -ev t -mor RF --percentage 100 


