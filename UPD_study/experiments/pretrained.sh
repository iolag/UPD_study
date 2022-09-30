# Activate the virtual environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lagi

# seed = 10

python UPD_study/models/AMCons/AMCtrainer.py -lp t -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py -lp t -mod MRI  
python UPD_study/models/DFR/DFRtrainer.py -lp t -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py -lp t -mod MRI  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t -mod MRI
python UPD_study/models/PADIM/PADIMtrainer.py -lp t -mod MRI  
python UPD_study/models/PII/PIItrainer.py -lp t -mod MRI  
python UPD_study/models/RD/RDtrainer.py -lp t -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py -lp t -mod MRI  

python UPD_study/models/AMCons/AMCtrainer.py -lp t -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py -lp t -mod MRI -seq t1  
python UPD_study/models/DFR/DFRtrainer.py -lp t -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py -lp t -mod MRI -seq t1  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t -mod MRI -seq t1
python UPD_study/models/PADIM/PADIMtrainer.py -lp t -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py -lp t -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py -lp t -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py -lp t -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py -lp t -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py -lp t -mod CXR  
python UPD_study/models/DFR/DFRtrainer.py -lp t -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py -lp t -mod CXR  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t -mod CXR
python UPD_study/models/PADIM/PADIMtrainer.py -lp t -mod CXR  
python UPD_study/models/PII/PIItrainer.py -lp t -mod CXR  
python UPD_study/models/RD/RDtrainer.py -lp t -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py -lp t -mod CXR  

python UPD_study/models/AMCons/AMCtrainer.py -lp t -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t -mod RF  
python UPD_study/models/DAE/DAEtrainer.py -lp t -mod RF  
python UPD_study/models/DFR/DFRtrainer.py -lp t -mod RF  
python UPD_study/models/FAE/FAEtrainer.py -lp t -mod RF  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t -mod RF
python UPD_study/models/PADIM/PADIMtrainer.py -lp t -mod RF  
python UPD_study/models/PII/PIItrainer.py -lp t -mod RF  
python UPD_study/models/RD/RDtrainer.py -lp t -mod RF  
python UPD_study/models/VAE/VAEtrainer.py -lp t -mod RF  
  
python UPD_study/models/AMCons/AMCtrainer.py -lp t -ev t -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t -ev t -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py -lp t -ev t -mod MRI  
python UPD_study/models/DFR/DFRtrainer.py -lp t -ev t -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py -lp t -ev t -mod MRI  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t -ev t -mod MRI  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t -ev t -mod MRI  
python UPD_study/models/PII/PIItrainer.py -lp t -ev t -mod MRI  
python UPD_study/models/RD/RDtrainer.py -lp t -ev t -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py -lp t -ev t -mod MRI  
python UPD_study/models/VAE/r-VAEtrainer.py -lp t -ev t -mod MRI 

python UPD_study/models/AMCons/AMCtrainer.py -lp t -ev t -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t -ev t -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py -lp t -ev t -mod MRI -seq t1  
python UPD_study/models/DFR/DFRtrainer.py -lp t -ev t -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py -lp t -ev t -mod MRI -seq t1  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t -ev t -mod MRI -seq t1  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t -ev t -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py -lp t -ev t -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py -lp t -ev t -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py -lp t -ev t -mod MRI -seq t1
python UPD_study/models/VAE/r-VAEtrainer.py -lp t -ev t -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/DAE/DAEtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/DFR/DFRtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/FAE/FAEtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/PII/PIItrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/RD/RDtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/VAE/VAEtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f
python UPD_study/models/VAE/r-VAEtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f

python UPD_study/models/AMCons/AMCtrainer.py -lp t -ev t -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t -ev t -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py -lp t -ev t -mod CXR  
python UPD_study/models/DFR/DFRtrainer.py -lp t -ev t -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py -lp t -ev t -mod CXR  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t -ev t -mod CXR  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t -ev t -mod CXR  
python UPD_study/models/PII/PIItrainer.py -lp t -ev t -mod CXR  
python UPD_study/models/RD/RDtrainer.py -lp t -ev t -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py -lp t -ev t -mod CXR  
python UPD_study/models/VAE/r-VAEtrainer.py -lp t -ev t -mod CXR

python UPD_study/models/AMCons/AMCtrainer.py -lp t -ev t -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t -ev t -mod RF  
python UPD_study/models/DAE/DAEtrainer.py -lp t -ev t -mod RF  
python UPD_study/models/DFR/DFRtrainer.py -lp t -ev t -mod RF  
python UPD_study/models/FAE/FAEtrainer.py -lp t -ev t -mod RF  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t -ev t -mod RF  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t -ev t -mod RF  
python UPD_study/models/PII/PIItrainer.py -lp t -ev t -mod RF  
python UPD_study/models/RD/RDtrainer.py -lp t -ev t -mod RF  
python UPD_study/models/VAE/VAEtrainer.py -lp t -ev t -mod RF  
python UPD_study/models/VAE/r-VAEtrainer.py -lp t -ev t -mod RF  

# seed = 20

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 20 -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 20 -mod MRI  
python UPD_study/models/DFR/DFRtrainer.py -lp t --seed 20 -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 20 -mod MRI  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t --seed 20 -mod MRI
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 20 -mod MRI  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 20 -mod MRI  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 20 -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 20 -mod MRI  

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 20 -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 20 -mod MRI -seq t1  
python UPD_study/models/DFR/DFRtrainer.py -lp t --seed 20 -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 20 -mod MRI -seq t1  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t --seed 20 -mod MRI -seq t1
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 20 -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 20 -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 20 -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 20 -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 20 -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 20 -mod CXR  
python UPD_study/models/DFR/DFRtrainer.py -lp t --seed 20 -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 20 -mod CXR  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t --seed 20 -mod CXR
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 20 -mod CXR  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 20 -mod CXR  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 20 -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 20 -mod CXR  

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 20 -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -mod RF  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 20 -mod RF  
python UPD_study/models/DFR/DFRtrainer.py -lp t --seed 20 -mod RF  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 20 -mod RF  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t --seed 20 -mod RF
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 20 -mod RF  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 20 -mod RF  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 20 -mod RF  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 20 -mod RF  
  
python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 20 -ev t -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -ev t -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 20 -ev t -mod MRI  
python UPD_study/models/DFR/DFRtrainer.py -lp t --seed 20 -ev t -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 20 -ev t -mod MRI  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t --seed 20 -ev t -mod MRI  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 20 -ev t -mod MRI  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 20 -ev t -mod MRI  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 20 -ev t -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 20 -ev t -mod MRI  
python UPD_study/models/VAE/r-VAEtrainer.py -lp t --seed 20 -ev t -mod MRI 

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/DFR/DFRtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1
python UPD_study/models/VAE/r-VAEtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/DFR/DFRtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f
python UPD_study/models/VAE/r-VAEtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 20 -ev t -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -ev t -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 20 -ev t -mod CXR  
python UPD_study/models/DFR/DFRtrainer.py -lp t --seed 20 -ev t -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 20 -ev t -mod CXR  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t --seed 20 -ev t -mod CXR  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 20 -ev t -mod CXR  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 20 -ev t -mod CXR  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 20 -ev t -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 20 -ev t -mod CXR  
python UPD_study/models/VAE/r-VAEtrainer.py -lp t --seed 20 -ev t -mod CXR

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 20 -ev t -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -ev t -mod RF  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 20 -ev t -mod RF  
python UPD_study/models/DFR/DFRtrainer.py -lp t --seed 20 -ev t -mod RF  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 20 -ev t -mod RF  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t --seed 20 -ev t -mod RF  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 20 -ev t -mod RF  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 20 -ev t -mod RF  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 20 -ev t -mod RF  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 20 -ev t -mod RF  
python UPD_study/models/VAE/r-VAEtrainer.py -lp t --seed 20 -ev t -mod RF  

# seed = 30
python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 30 -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 30 -mod MRI  
python UPD_study/models/DFR/DFRtrainer.py -lp t --seed 30 -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 30 -mod MRI  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t --seed 30 -mod MRI
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 30 -mod MRI  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 30 -mod MRI  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 30 -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 30 -mod MRI  

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 30 -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 30 -mod MRI -seq t1  
python UPD_study/models/DFR/DFRtrainer.py -lp t --seed 30 -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 30 -mod MRI -seq t1  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t --seed 30 -mod MRI -seq t1
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 30 -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 30 -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 30 -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 30 -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 30 -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 30 -mod CXR  
python UPD_study/models/DFR/DFRtrainer.py -lp t --seed 30 -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 30 -mod CXR  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t --seed 30 -mod CXR
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 30 -mod CXR  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 30 -mod CXR  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 30 -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 30 -mod CXR  

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 30 -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -mod RF  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 30 -mod RF  
python UPD_study/models/DFR/DFRtrainer.py -lp t --seed 30 -mod RF  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 30 -mod RF  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t --seed 30 -mod RF
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 30 -mod RF  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 30 -mod RF  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 30 -mod RF  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 30 -mod RF  
  
python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 30 -ev t -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -ev t -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 30 -ev t -mod MRI  
python UPD_study/models/DFR/DFRtrainer.py -lp t --seed 30 -ev t -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 30 -ev t -mod MRI  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t --seed 30 -ev t -mod MRI  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 30 -ev t -mod MRI  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 30 -ev t -mod MRI  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 30 -ev t -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 30 -ev t -mod MRI  
python UPD_study/models/VAE/r-VAEtrainer.py -lp t --seed 30 -ev t -mod MRI 

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/DFR/DFRtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1
python UPD_study/models/VAE/r-VAEtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/DFR/DFRtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f
python UPD_study/models/VAE/r-VAEtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 30 -ev t -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -ev t -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 30 -ev t -mod CXR  
python UPD_study/models/DFR/DFRtrainer.py -lp t --seed 30 -ev t -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 30 -ev t -mod CXR  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t --seed 30 -ev t -mod CXR  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 30 -ev t -mod CXR  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 30 -ev t -mod CXR  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 30 -ev t -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 30 -ev t -mod CXR  
python UPD_study/models/VAE/r-VAEtrainer.py -lp t --seed 30 -ev t -mod CXR

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 30 -ev t -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -ev t -mod RF  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 30 -ev t -mod RF  
python UPD_study/models/DFR/DFRtrainer.py -lp t --seed 30 -ev t -mod RF  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 30 -ev t -mod RF  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t --seed 30 -ev t -mod RF  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 30 -ev t -mod RF  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 30 -ev t -mod RF  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 30 -ev t -mod RF  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 30 -ev t -mod RF  
python UPD_study/models/VAE/r-VAEtrainer.py -lp t --seed 30 -ev t -mod RF  

