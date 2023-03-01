# # Activate the virtual environment
# source /opt/anaconda3/etc/profile.d/conda.sh
# conda activate upd

# Method training

python UPD_study/models/AMCons/AMCtrainer.py -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py -mod MRI  
python UPD_study/models/fAnoGAN/GANtrainer.py -mod MRI
python UPD_study/models/PADIM/PADIMtrainer.py -mod MRI  
python UPD_study/models/PII/PIItrainer.py -mod MRI  
python UPD_study/models/RD/RDtrainer.py -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py -mod MRI  

python UPD_study/models/AMCons/AMCtrainer.py -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py -mod MRI -seq t1  
python UPD_study/models/fAnoGAN/GANtrainer.py -mod MRI -seq t1
python UPD_study/models/PADIM/PADIMtrainer.py -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py -mod CXR  
python UPD_study/models/fAnoGAN/GANtrainer.py -mod CXR
python UPD_study/models/PADIM/PADIMtrainer.py -mod CXR  
python UPD_study/models/PII/PIItrainer.py -mod CXR  
python UPD_study/models/RD/RDtrainer.py -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py -mod CXR  

python UPD_study/models/AMCons/AMCtrainer.py -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py -mod RF  
python UPD_study/models/DAE/DAEtrainer.py -mod RF  
python UPD_study/models/FAE/FAEtrainer.py -mod RF  
python UPD_study/models/fAnoGAN/GANtrainer.py -mod RF
python UPD_study/models/PADIM/PADIMtrainer.py -mod RF  
python UPD_study/models/PII/PIItrainer.py -mod RF  
python UPD_study/models/RD/RDtrainer.py -mod RF  
python UPD_study/models/VAE/VAEtrainer.py -mod RF  

# Method evaluation

python UPD_study/models/AMCons/AMCtrainer.py -ev t -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py -ev t -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py -ev t -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py -ev t -mod MRI  
python UPD_study/models/fAnoGAN/GANtrainer.py -ev t -mod MRI  
python UPD_study/models/PADIM/PADIMtrainer.py -ev t -mod MRI  
python UPD_study/models/PII/PIItrainer.py -ev t -mod MRI  
python UPD_study/models/RD/RDtrainer.py -ev t -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py -ev t -mod MRI  
python UPD_study/models/VAE/r-VAEtrainer.py -ev t -mod MRI 

python UPD_study/models/AMCons/AMCtrainer.py -ev t -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py -ev t -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py -ev t -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py -ev t -mod MRI -seq t1  
python UPD_study/models/fAnoGAN/GANtrainer.py -ev t -mod MRI -seq t1  
python UPD_study/models/PADIM/PADIMtrainer.py -ev t -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py -ev t -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py -ev t -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py -ev t -mod MRI -seq t1
python UPD_study/models/VAE/r-VAEtrainer.py -ev t -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py -ev t -mod MRI -seq t1 --brats_t1 f 
python UPD_study/models/CFLOW-AD/CFtrainer.py -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/DAE/DAEtrainer.py -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/FAE/FAEtrainer.py -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/fAnoGAN/GANtrainer.py -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/PADIM/PADIMtrainer.py -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/PII/PIItrainer.py -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/RD/RDtrainer.py -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/VAE/VAEtrainer.py -ev t -mod MRI -seq t1 --brats_t1 f
python UPD_study/models/VAE/r-VAEtrainer.py -ev t -mod MRI -seq t1 --brats_t1 f

python UPD_study/models/AMCons/AMCtrainer.py -ev t -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py -ev t -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py -ev t -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py -ev t -mod CXR  
python UPD_study/models/fAnoGAN/GANtrainer.py -ev t -mod CXR  
python UPD_study/models/PADIM/PADIMtrainer.py -ev t -mod CXR  
python UPD_study/models/PII/PIItrainer.py -ev t -mod CXR  
python UPD_study/models/RD/RDtrainer.py -ev t -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py -ev t -mod CXR  
python UPD_study/models/VAE/r-VAEtrainer.py -ev t -mod CXR

python UPD_study/models/AMCons/AMCtrainer.py -ev t -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py -ev t -mod RF  
python UPD_study/models/DAE/DAEtrainer.py -ev t -mod RF  
python UPD_study/models/FAE/FAEtrainer.py -ev t -mod RF  
python UPD_study/models/fAnoGAN/GANtrainer.py -ev t -mod RF  
python UPD_study/models/PADIM/PADIMtrainer.py -ev t -mod RF  
python UPD_study/models/PII/PIItrainer.py -ev t -mod RF  
python UPD_study/models/RD/RDtrainer.py -ev t -mod RF  
python UPD_study/models/VAE/VAEtrainer.py -ev t -mod RF  
python UPD_study/models/VAE/r-VAEtrainer.py -ev t -mod RF  

# seed = 20

python UPD_study/models/AMCons/AMCtrainer.py --seed 20 -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 20 -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py --seed 20 -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py --seed 20 -mod MRI  
python UPD_study/models/fAnoGAN/GANtrainer.py --seed 20 -mod MRI
python UPD_study/models/PADIM/PADIMtrainer.py --seed 20 -mod MRI  
python UPD_study/models/PII/PIItrainer.py --seed 20 -mod MRI  
python UPD_study/models/RD/RDtrainer.py --seed 20 -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py --seed 20 -mod MRI  

python UPD_study/models/AMCons/AMCtrainer.py --seed 20 -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 20 -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py --seed 20 -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py --seed 20 -mod MRI -seq t1  
python UPD_study/models/fAnoGAN/GANtrainer.py --seed 20 -mod MRI -seq t1
python UPD_study/models/PADIM/PADIMtrainer.py --seed 20 -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py --seed 20 -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py --seed 20 -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py --seed 20 -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py --seed 20 -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 20 -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py --seed 20 -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py --seed 20 -mod CXR  
python UPD_study/models/fAnoGAN/GANtrainer.py --seed 20 -mod CXR
python UPD_study/models/PADIM/PADIMtrainer.py --seed 20 -mod CXR  
python UPD_study/models/PII/PIItrainer.py --seed 20 -mod CXR  
python UPD_study/models/RD/RDtrainer.py --seed 20 -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py --seed 20 -mod CXR  

python UPD_study/models/AMCons/AMCtrainer.py --seed 20 -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 20 -mod RF  
python UPD_study/models/DAE/DAEtrainer.py --seed 20 -mod RF  
python UPD_study/models/FAE/FAEtrainer.py --seed 20 -mod RF  
python UPD_study/models/fAnoGAN/GANtrainer.py --seed 20 -mod RF
python UPD_study/models/PADIM/PADIMtrainer.py --seed 20 -mod RF  
python UPD_study/models/PII/PIItrainer.py --seed 20 -mod RF  
python UPD_study/models/RD/RDtrainer.py --seed 20 -mod RF  
python UPD_study/models/VAE/VAEtrainer.py --seed 20 -mod RF  
  
python UPD_study/models/AMCons/AMCtrainer.py --seed 20 -ev t -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 20 -ev t -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py --seed 20 -ev t -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py --seed 20 -ev t -mod MRI  
python UPD_study/models/fAnoGAN/GANtrainer.py --seed 20 -ev t -mod MRI  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 20 -ev t -mod MRI  
python UPD_study/models/PII/PIItrainer.py --seed 20 -ev t -mod MRI  
python UPD_study/models/RD/RDtrainer.py --seed 20 -ev t -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py --seed 20 -ev t -mod MRI  
python UPD_study/models/VAE/r-VAEtrainer.py --seed 20 -ev t -mod MRI 

python UPD_study/models/AMCons/AMCtrainer.py --seed 20 -ev t -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/fAnoGAN/GANtrainer.py --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py --seed 20 -ev t -mod MRI -seq t1
python UPD_study/models/VAE/r-VAEtrainer.py --seed 20 -ev t -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/DAE/DAEtrainer.py --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/FAE/FAEtrainer.py --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/fAnoGAN/GANtrainer.py --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/PII/PIItrainer.py --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/RD/RDtrainer.py --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/VAE/VAEtrainer.py --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f
python UPD_study/models/VAE/r-VAEtrainer.py --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f

python UPD_study/models/AMCons/AMCtrainer.py --seed 20 -ev t -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 20 -ev t -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py --seed 20 -ev t -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py --seed 20 -ev t -mod CXR  
python UPD_study/models/fAnoGAN/GANtrainer.py --seed 20 -ev t -mod CXR  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 20 -ev t -mod CXR  
python UPD_study/models/PII/PIItrainer.py --seed 20 -ev t -mod CXR  
python UPD_study/models/RD/RDtrainer.py --seed 20 -ev t -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py --seed 20 -ev t -mod CXR  
python UPD_study/models/VAE/r-VAEtrainer.py --seed 20 -ev t -mod CXR

python UPD_study/models/AMCons/AMCtrainer.py --seed 20 -ev t -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 20 -ev t -mod RF  
python UPD_study/models/DAE/DAEtrainer.py --seed 20 -ev t -mod RF  
python UPD_study/models/FAE/FAEtrainer.py --seed 20 -ev t -mod RF  
python UPD_study/models/fAnoGAN/GANtrainer.py --seed 20 -ev t -mod RF  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 20 -ev t -mod RF  
python UPD_study/models/PII/PIItrainer.py --seed 20 -ev t -mod RF  
python UPD_study/models/RD/RDtrainer.py --seed 20 -ev t -mod RF  
python UPD_study/models/VAE/VAEtrainer.py --seed 20 -ev t -mod RF  
python UPD_study/models/VAE/r-VAEtrainer.py --seed 20 -ev t -mod RF  

# seed = 30
python UPD_study/models/AMCons/AMCtrainer.py --seed 30 -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 30 -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py --seed 30 -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py --seed 30 -mod MRI  
python UPD_study/models/fAnoGAN/GANtrainer.py --seed 30 -mod MRI
python UPD_study/models/PADIM/PADIMtrainer.py --seed 30 -mod MRI  
python UPD_study/models/PII/PIItrainer.py --seed 30 -mod MRI  
python UPD_study/models/RD/RDtrainer.py --seed 30 -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py --seed 30 -mod MRI  

python UPD_study/models/AMCons/AMCtrainer.py --seed 30 -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 30 -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py --seed 30 -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py --seed 30 -mod MRI -seq t1  
python UPD_study/models/fAnoGAN/GANtrainer.py --seed 30 -mod MRI -seq t1
python UPD_study/models/PADIM/PADIMtrainer.py --seed 30 -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py --seed 30 -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py --seed 30 -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py --seed 30 -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py --seed 30 -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 30 -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py --seed 30 -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py --seed 30 -mod CXR  
python UPD_study/models/fAnoGAN/GANtrainer.py --seed 30 -mod CXR
python UPD_study/models/PADIM/PADIMtrainer.py --seed 30 -mod CXR  
python UPD_study/models/PII/PIItrainer.py --seed 30 -mod CXR  
python UPD_study/models/RD/RDtrainer.py --seed 30 -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py --seed 30 -mod CXR  

python UPD_study/models/AMCons/AMCtrainer.py --seed 30 -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 30 -mod RF  
python UPD_study/models/DAE/DAEtrainer.py --seed 30 -mod RF  
python UPD_study/models/FAE/FAEtrainer.py --seed 30 -mod RF  
python UPD_study/models/fAnoGAN/GANtrainer.py --seed 30 -mod RF
python UPD_study/models/PADIM/PADIMtrainer.py --seed 30 -mod RF  
python UPD_study/models/PII/PIItrainer.py --seed 30 -mod RF  
python UPD_study/models/RD/RDtrainer.py --seed 30 -mod RF  
python UPD_study/models/VAE/VAEtrainer.py --seed 30 -mod RF  
  
python UPD_study/models/AMCons/AMCtrainer.py --seed 30 -ev t -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 30 -ev t -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py --seed 30 -ev t -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py --seed 30 -ev t -mod MRI  
python UPD_study/models/fAnoGAN/GANtrainer.py --seed 30 -ev t -mod MRI  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 30 -ev t -mod MRI  
python UPD_study/models/PII/PIItrainer.py --seed 30 -ev t -mod MRI  
python UPD_study/models/RD/RDtrainer.py --seed 30 -ev t -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py --seed 30 -ev t -mod MRI  
python UPD_study/models/VAE/r-VAEtrainer.py --seed 30 -ev t -mod MRI 

python UPD_study/models/AMCons/AMCtrainer.py --seed 30 -ev t -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/fAnoGAN/GANtrainer.py --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py --seed 30 -ev t -mod MRI -seq t1
python UPD_study/models/VAE/r-VAEtrainer.py --seed 30 -ev t -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/DAE/DAEtrainer.py --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/FAE/FAEtrainer.py --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/fAnoGAN/GANtrainer.py --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/PII/PIItrainer.py --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/RD/RDtrainer.py --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/VAE/VAEtrainer.py --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f
python UPD_study/models/VAE/r-VAEtrainer.py --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f

python UPD_study/models/AMCons/AMCtrainer.py --seed 30 -ev t -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 30 -ev t -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py --seed 30 -ev t -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py --seed 30 -ev t -mod CXR  
python UPD_study/models/fAnoGAN/GANtrainer.py --seed 30 -ev t -mod CXR  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 30 -ev t -mod CXR  
python UPD_study/models/PII/PIItrainer.py --seed 30 -ev t -mod CXR  
python UPD_study/models/RD/RDtrainer.py --seed 30 -ev t -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py --seed 30 -ev t -mod CXR  
python UPD_study/models/VAE/r-VAEtrainer.py --seed 30 -ev t -mod CXR

python UPD_study/models/AMCons/AMCtrainer.py --seed 30 -ev t -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 30 -ev t -mod RF  
python UPD_study/models/DAE/DAEtrainer.py --seed 30 -ev t -mod RF  
python UPD_study/models/FAE/FAEtrainer.py --seed 30 -ev t -mod RF  
python UPD_study/models/fAnoGAN/GANtrainer.py --seed 30 -ev t -mod RF  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 30 -ev t -mod RF  
python UPD_study/models/PII/PIItrainer.py --seed 30 -ev t -mod RF  
python UPD_study/models/RD/RDtrainer.py --seed 30 -ev t -mod RF  
python UPD_study/models/VAE/VAEtrainer.py --seed 30 -ev t -mod RF  
python UPD_study/models/VAE/r-VAEtrainer.py --seed 30 -ev t -mod RF  



#newly added methods: expVAE, CutPaste

#expVAE

python UPD_study/models/expVAE/expVAEtrainer.py  -mod MRI --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py  -mod MRI --seed 20 
python UPD_study/models/expVAE/expVAEtrainer.py  -mod MRI --seed 30
python UPD_study/models/expVAE/expVAEtrainer.py  -mod MRI -seq t1 --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py  -mod MRI -seq t1 --seed 20 
python UPD_study/models/expVAE/expVAEtrainer.py  -mod MRI -seq t1 --seed 30
python UPD_study/models/expVAE/expVAEtrainer.py  -mod MRI --brats_t1 f -seq t1 --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py  -mod MRI --brats_t1 f -seq t1 --seed 20 
python UPD_study/models/expVAE/expVAEtrainer.py  -mod MRI --brats_t1 f -seq t1 --seed 30
python UPD_study/models/expVAE/expVAEtrainer.py  -mod RF --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py  -mod RF --seed 20 
python UPD_study/models/expVAE/expVAEtrainer.py  -mod RF --seed 30
python UPD_study/models/expVAE/expVAEtrainer.py  -mod CXR --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py  -mod CXR --seed 20 
python UPD_study/models/expVAE/expVAEtrainer.py  -mod CXR --seed 30

python UPD_study/models/expVAE/expVAEtrainer.py -ev t -mod MRI --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py -ev t  -mod MRI --seed 20 
python UPD_study/models/expVAE/expVAEtrainer.py -ev t  -mod MRI --seed 30
python UPD_study/models/expVAE/expVAEtrainer.py -ev t  -mod MRI -seq t1 --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py -ev t  -mod MRI -seq t1 --seed 20 
python UPD_study/models/expVAE/expVAEtrainer.py -ev t  -mod MRI -seq t1 --seed 30
python UPD_study/models/expVAE/expVAEtrainer.py -ev t  -mod MRI --brats_t1 f -seq t1 --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py -ev t  -mod MRI --brats_t1 f -seq t1 --seed 20 
python UPD_study/models/expVAE/expVAEtrainer.py -ev t  -mod MRI --brats_t1 f -seq t1 --seed 30
python UPD_study/models/expVAE/expVAEtrainer.py -ev t  -mod RF --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py -ev t  -mod RF --seed 20 
python UPD_study/models/expVAE/expVAEtrainer.py -ev t  -mod RF --seed 30
python UPD_study/models/expVAE/expVAEtrainer.py -ev t  -mod CXR --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py -ev t  -mod CXR --seed 20 
python UPD_study/models/expVAE/expVAEtrainer.py -ev t  -mod CXR --seed 30


#CutPaste: localization
python UPD_study/models/CutPaste/CPtrainer.py  -mod MRI --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py  -mod MRI --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py  -mod MRI --seed 30
python UPD_study/models/CutPaste/CPtrainer.py  -mod MRI -seq t1 --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py  -mod MRI -seq t1 --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py  -mod MRI -seq t1 --seed 30
python UPD_study/models/CutPaste/CPtrainer.py  -mod MRI --brats_t1 f -seq t1 --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py  -mod MRI --brats_t1 f -seq t1 --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py  -mod MRI --brats_t1 f -seq t1 --seed 30
python UPD_study/models/CutPaste/CPtrainer.py  -mod RF --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py  -mod RF --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py  -mod RF --seed 30
python UPD_study/models/CutPaste/CPtrainer.py  -mod CXR --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py  -mod CXR --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py  -mod CXR --seed 30

python UPD_study/models/CutPaste/CPtrainer.py -ev t -mod MRI --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -ev t  -mod MRI --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -ev t  -mod MRI --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -ev t  -mod MRI -seq t1 --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -ev t  -mod MRI -seq t1 --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -ev t  -mod MRI -seq t1 --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -ev t  -mod MRI --brats_t1 f -seq t1 --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -ev t  -mod MRI --brats_t1 f -seq t1 --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -ev t  -mod MRI --brats_t1 f -seq t1 --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -ev t  -mod RF --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -ev t  -mod RF --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -ev t  -mod RF --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -ev t  -mod CXR --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -ev t  -mod CXR --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -ev t  -mod CXR --seed 30

#CutPaste: image level
python UPD_study/models/CutPaste/CPtrainer.py -loc f  -mod MRI --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -loc f  -mod MRI --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -loc f  -mod MRI --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -loc f  -mod MRI -seq t1 --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -loc f  -mod MRI -seq t1 --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -loc f  -mod MRI -seq t1 --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -loc f  -mod MRI --brats_t1 f -seq t1 --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -loc f  -mod MRI --brats_t1 f -seq t1 --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -loc f  -mod MRI --brats_t1 f -seq t1 --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -loc f  -mod RF --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -loc f  -mod RF --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -loc f  -mod RF --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -loc f  -mod CXR --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -loc f  -mod CXR --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -loc f  -mod CXR --seed 30

python UPD_study/models/CutPaste/CPtrainer.py -loc f -ev t -mod MRI --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -loc f -ev t  -mod MRI --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -loc f -ev t  -mod MRI --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -loc f -ev t  -mod MRI -seq t1 --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -loc f -ev t  -mod MRI -seq t1 --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -loc f -ev t  -mod MRI -seq t1 --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -loc f -ev t  -mod MRI --brats_t1 f -seq t1 --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -loc f -ev t  -mod MRI --brats_t1 f -seq t1 --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -loc f -ev t  -mod MRI --brats_t1 f -seq t1 --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -loc f -ev t  -mod RF --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -loc f -ev t  -mod RF --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -loc f -ev t  -mod RF --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -loc f -ev t  -mod CXR --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -loc f -ev t  -mod CXR --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -loc f -ev t  -mod CXR --seed 30

#H-TAE-S

python UPD_study/models/H-TAE-S/HTAEStrainer.py  -mod MRI --seed 10 
python UPD_study/models/H-TAE-S/HTAEStrainer.py  -mod MRI --seed 20 
python UPD_study/models/H-TAE-S/HTAEStrainer.py  -mod MRI --seed 30
python UPD_study/models/H-TAE-S/HTAEStrainer.py  -mod MRI -seq t1 --seed 10 
python UPD_study/models/H-TAE-S/HTAEStrainer.py  -mod MRI -seq t1 --seed 20 
python UPD_study/models/H-TAE-S/HTAEStrainer.py  -mod MRI -seq t1 --seed 30
python UPD_study/models/H-TAE-S/HTAEStrainer.py  -mod MRI --brats_t1 f -seq t1 --seed 10 
python UPD_study/models/H-TAE-S/HTAEStrainer.py  -mod MRI --brats_t1 f -seq t1 --seed 20 
python UPD_study/models/H-TAE-S/HTAEStrainer.py  -mod MRI --brats_t1 f -seq t1 --seed 30
python UPD_study/models/H-TAE-S/HTAEStrainer.py  -mod RF --seed 10 
python UPD_study/models/H-TAE-S/HTAEStrainer.py  -mod RF --seed 20 
python UPD_study/models/H-TAE-S/HTAEStrainer.py  -mod RF --seed 30
python UPD_study/models/H-TAE-S/HTAEStrainer.py  -mod CXR --seed 10 
python UPD_study/models/H-TAE-S/HTAEStrainer.py  -mod CXR --seed 20 
python UPD_study/models/H-TAE-S/HTAEStrainer.py  -mod CXR --seed 30

python UPD_study/models/H-TAE-S/HTAEStrainer.py -ev t -mod MRI --seed 10 
python UPD_study/models/H-TAE-S/HTAEStrainer.py -ev t  -mod MRI --seed 20 
python UPD_study/models/H-TAE-S/HTAEStrainer.py -ev t  -mod MRI --seed 30
python UPD_study/models/H-TAE-S/HTAEStrainer.py -ev t  -mod MRI -seq t1 --seed 10 
python UPD_study/models/H-TAE-S/HTAEStrainer.py -ev t  -mod MRI -seq t1 --seed 20 
python UPD_study/models/H-TAE-S/HTAEStrainer.py -ev t  -mod MRI -seq t1 --seed 30
python UPD_study/models/H-TAE-S/HTAEStrainer.py -ev t  -mod MRI --brats_t1 f -seq t1 --seed 10 
python UPD_study/models/H-TAE-S/HTAEStrainer.py -ev t  -mod MRI --brats_t1 f -seq t1 --seed 20 
python UPD_study/models/H-TAE-S/HTAEStrainer.py -ev t  -mod MRI --brats_t1 f -seq t1 --seed 30
python UPD_study/models/H-TAE-S/HTAEStrainer.py -ev t  -mod RF --seed 10 
python UPD_study/models/H-TAE-S/HTAEStrainer.py -ev t  -mod RF --seed 20 
python UPD_study/models/H-TAE-S/HTAEStrainer.py -ev t  -mod RF --seed 30
python UPD_study/models/H-TAE-S/HTAEStrainer.py -ev t  -mod CXR --seed 10 
python UPD_study/models/H-TAE-S/HTAEStrainer.py -ev t  -mod CXR --seed 20 
python UPD_study/models/H-TAE-S/HTAEStrainer.py -ev t  -mod CXR --seed 30