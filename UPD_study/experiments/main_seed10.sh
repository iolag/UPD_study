# Activate the virtual environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lagi

# Method training

python models/AMCons/AMCtrainer.py -mod MRI 
python models/CFLOW-AD/CFtrainer.py -mod MRI  
python models/DAE/DAEtrainer.py -mod MRI  
python models/DFR/DFRtrainer.py -mod MRI  
python models/FAE/FAEtrainer.py -mod MRI  
python models/f-AnoGAN/GANtrainer.py -mod MRI
python models/PADIM/PADIMtrainer.py -mod MRI  
python models/PII/PIItrainer.py -mod MRI  
python models/RD/RDtrainer.py -mod MRI  
python models/VAE/VAEtrainer.py -mod MRI  

python models/AMCons/AMCtrainer.py -mod MRI -seq t1 
python models/CFLOW-AD/CFtrainer.py -mod MRI -seq t1  
python models/DAE/DAEtrainer.py -mod MRI -seq t1  
python models/DFR/DFRtrainer.py -mod MRI -seq t1  
python models/FAE/FAEtrainer.py -mod MRI -seq t1  
python models/f-AnoGAN/GANtrainer.py -mod MRI -seq t1
python models/PADIM/PADIMtrainer.py -mod MRI -seq t1  
python models/PII/PIItrainer.py -mod MRI -seq t1  
python models/RD/RDtrainer.py -mod MRI -seq t1  
python models/VAE/VAEtrainer.py -mod MRI -seq t1

python models/AMCons/AMCtrainer.py -mod CXR 
python models/CFLOW-AD/CFtrainer.py -mod CXR  
python models/DAE/DAEtrainer.py -mod CXR  
python models/DFR/DFRtrainer.py -mod CXR  
python models/FAE/FAEtrainer.py -mod CXR  
python models/f-AnoGAN/GANtrainer.py -mod CXR
python models/PADIM/PADIMtrainer.py -mod CXR  
python models/PII/PIItrainer.py -mod CXR  
python models/RD/RDtrainer.py -mod CXR  
python models/VAE/VAEtrainer.py -mod CXR  

python models/AMCons/AMCtrainer.py -mod RF 
python models/CFLOW-AD/CFtrainer.py -mod RF  
python models/DAE/DAEtrainer.py -mod RF  
python models/DFR/DFRtrainer.py -mod RF  
python models/FAE/FAEtrainer.py -mod RF  
python models/f-AnoGAN/GANtrainer.py -mod RF
python models/PADIM/PADIMtrainer.py -mod RF  
python models/PII/PIItrainer.py -mod RF  
python models/RD/RDtrainer.py -mod RF  
python models/VAE/VAEtrainer.py -mod RF  

# Method evaluation

python models/AMCons/AMCtrainer.py -ev t -mod MRI 
python models/CFLOW-AD/CFtrainer.py -ev t -mod MRI  
python models/DAE/DAEtrainer.py -ev t -mod MRI  
python models/DFR/DFRtrainer.py -ev t -mod MRI  
python models/FAE/FAEtrainer.py -ev t -mod MRI  
python models/f-AnoGAN/GANtrainer.py -ev t -mod MRI  
python models/PADIM/PADIMtrainer.py -ev t -mod MRI  
python models/PII/PIItrainer.py -ev t -mod MRI  
python models/RD/RDtrainer.py -ev t -mod MRI  
python models/VAE/VAEtrainer.py -ev t -mod MRI  
python models/VAE/r-VAEtrainer.py -ev t -mod MRI 

python models/AMCons/AMCtrainer.py -ev t -mod MRI -seq t1 
python models/CFLOW-AD/CFtrainer.py -ev t -mod MRI -seq t1  
python models/DAE/DAEtrainer.py -ev t -mod MRI -seq t1  
python models/DFR/DFRtrainer.py -ev t -mod MRI -seq t1  
python models/FAE/FAEtrainer.py -ev t -mod MRI -seq t1  
python models/f-AnoGAN/GANtrainer.py -ev t -mod MRI -seq t1  
python models/PADIM/PADIMtrainer.py -ev t -mod MRI -seq t1  
python models/PII/PIItrainer.py -ev t -mod MRI -seq t1  
python models/RD/RDtrainer.py -ev t -mod MRI -seq t1  
python models/VAE/VAEtrainer.py -ev t -mod MRI -seq t1
python models/VAE/r-VAEtrainer.py -ev t -mod MRI -seq t1

python models/AMCons/AMCtrainer.py -ev t -mod MRI -seq t1 --brats_t1 f 
python models/CFLOW-AD/CFtrainer.py -ev t -mod MRI -seq t1 --brats_t1 f  
python models/DAE/DAEtrainer.py -ev t -mod MRI -seq t1 --brats_t1 f  
python models/DFR/DFRtrainer.py -ev t -mod MRI -seq t1 --brats_t1 f  
python models/FAE/FAEtrainer.py -ev t -mod MRI -seq t1 --brats_t1 f  
python models/f-AnoGAN/GANtrainer.py -ev t -mod MRI -seq t1 --brats_t1 f  
python models/PADIM/PADIMtrainer.py -ev t -mod MRI -seq t1 --brats_t1 f  
python models/PII/PIItrainer.py -ev t -mod MRI -seq t1 --brats_t1 f  
python models/RD/RDtrainer.py -ev t -mod MRI -seq t1 --brats_t1 f  
python models/VAE/VAEtrainer.py -ev t -mod MRI -seq t1 --brats_t1 f
python models/VAE/r-VAEtrainer.py -ev t -mod MRI -seq t1 --brats_t1 f

python models/AMCons/AMCtrainer.py -ev t -mod CXR 
python models/CFLOW-AD/CFtrainer.py -ev t -mod CXR  
python models/DAE/DAEtrainer.py -ev t -mod CXR  
python models/DFR/DFRtrainer.py -ev t -mod CXR  
python models/FAE/FAEtrainer.py -ev t -mod CXR  
python models/f-AnoGAN/GANtrainer.py -ev t -mod CXR  
python models/PADIM/PADIMtrainer.py -ev t -mod CXR  
python models/PII/PIItrainer.py -ev t -mod CXR  
python models/RD/RDtrainer.py -ev t -mod CXR  
python models/VAE/VAEtrainer.py -ev t -mod CXR  
python models/VAE/r-VAEtrainer.py -ev t -mod CXR

python models/AMCons/AMCtrainer.py -ev t -mod RF 
python models/CFLOW-AD/CFtrainer.py -ev t -mod RF  
python models/DAE/DAEtrainer.py -ev t -mod RF  
python models/DFR/DFRtrainer.py -ev t -mod RF  
python models/FAE/FAEtrainer.py -ev t -mod RF  
python models/f-AnoGAN/GANtrainer.py -ev t -mod RF  
python models/PADIM/PADIMtrainer.py -ev t -mod RF  
python models/PII/PIItrainer.py -ev t -mod RF  
python models/RD/RDtrainer.py -ev t -mod RF  
python models/VAE/VAEtrainer.py -ev t -mod RF  
python models/VAE/r-VAEtrainer.py -ev t -mod RF  

