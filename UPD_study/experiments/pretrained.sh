# Activate the virtual environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lagi

# seed = 10

python models/AMCons/AMCtrainer.py -lp t -mod MRI 
python models/CFLOW-AD/CFtrainer.py -lp t -mod MRI  
python models/DAE/DAEtrainer.py -lp t -mod MRI  
python models/DFR/DFRtrainer.py -lp t -mod MRI  
python models/FAE/FAEtrainer.py -lp t -mod MRI  
python models/f-AnoGAN/GANtrainer.py -lp t -mod MRI
python models/PADIM/PADIMtrainer.py -lp t -mod MRI  
python models/PII/PIItrainer.py -lp t -mod MRI  
python models/RD/RDtrainer.py -lp t -mod MRI  
python models/VAE/VAEtrainer.py -lp t -mod MRI  

python models/AMCons/AMCtrainer.py -lp t -mod MRI -seq t1 
python models/CFLOW-AD/CFtrainer.py -lp t -mod MRI -seq t1  
python models/DAE/DAEtrainer.py -lp t -mod MRI -seq t1  
python models/DFR/DFRtrainer.py -lp t -mod MRI -seq t1  
python models/FAE/FAEtrainer.py -lp t -mod MRI -seq t1  
python models/f-AnoGAN/GANtrainer.py -lp t -mod MRI -seq t1
python models/PADIM/PADIMtrainer.py -lp t -mod MRI -seq t1  
python models/PII/PIItrainer.py -lp t -mod MRI -seq t1  
python models/RD/RDtrainer.py -lp t -mod MRI -seq t1  
python models/VAE/VAEtrainer.py -lp t -mod MRI -seq t1

python models/AMCons/AMCtrainer.py -lp t -mod CXR 
python models/CFLOW-AD/CFtrainer.py -lp t -mod CXR  
python models/DAE/DAEtrainer.py -lp t -mod CXR  
python models/DFR/DFRtrainer.py -lp t -mod CXR  
python models/FAE/FAEtrainer.py -lp t -mod CXR  
python models/f-AnoGAN/GANtrainer.py -lp t -mod CXR
python models/PADIM/PADIMtrainer.py -lp t -mod CXR  
python models/PII/PIItrainer.py -lp t -mod CXR  
python models/RD/RDtrainer.py -lp t -mod CXR  
python models/VAE/VAEtrainer.py -lp t -mod CXR  

python models/AMCons/AMCtrainer.py -lp t -mod RF 
python models/CFLOW-AD/CFtrainer.py -lp t -mod RF  
python models/DAE/DAEtrainer.py -lp t -mod RF  
python models/DFR/DFRtrainer.py -lp t -mod RF  
python models/FAE/FAEtrainer.py -lp t -mod RF  
python models/f-AnoGAN/GANtrainer.py -lp t -mod RF
python models/PADIM/PADIMtrainer.py -lp t -mod RF  
python models/PII/PIItrainer.py -lp t -mod RF  
python models/RD/RDtrainer.py -lp t -mod RF  
python models/VAE/VAEtrainer.py -lp t -mod RF  
  
python models/AMCons/AMCtrainer.py -lp t -ev t -mod MRI 
python models/CFLOW-AD/CFtrainer.py -lp t -ev t -mod MRI  
python models/DAE/DAEtrainer.py -lp t -ev t -mod MRI  
python models/DFR/DFRtrainer.py -lp t -ev t -mod MRI  
python models/FAE/FAEtrainer.py -lp t -ev t -mod MRI  
python models/f-AnoGAN/GANtrainer.py -lp t -ev t -mod MRI  
python models/PADIM/PADIMtrainer.py -lp t -ev t -mod MRI  
python models/PII/PIItrainer.py -lp t -ev t -mod MRI  
python models/RD/RDtrainer.py -lp t -ev t -mod MRI  
python models/VAE/VAEtrainer.py -lp t -ev t -mod MRI  
python models/VAE/r-VAEtrainer.py -lp t -ev t -mod MRI 

python models/AMCons/AMCtrainer.py -lp t -ev t -mod MRI -seq t1 
python models/CFLOW-AD/CFtrainer.py -lp t -ev t -mod MRI -seq t1  
python models/DAE/DAEtrainer.py -lp t -ev t -mod MRI -seq t1  
python models/DFR/DFRtrainer.py -lp t -ev t -mod MRI -seq t1  
python models/FAE/FAEtrainer.py -lp t -ev t -mod MRI -seq t1  
python models/f-AnoGAN/GANtrainer.py -lp t -ev t -mod MRI -seq t1  
python models/PADIM/PADIMtrainer.py -lp t -ev t -mod MRI -seq t1  
python models/PII/PIItrainer.py -lp t -ev t -mod MRI -seq t1  
python models/RD/RDtrainer.py -lp t -ev t -mod MRI -seq t1  
python models/VAE/VAEtrainer.py -lp t -ev t -mod MRI -seq t1
python models/VAE/r-VAEtrainer.py -lp t -ev t -mod MRI -seq t1

python models/AMCons/AMCtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f 
python models/CFLOW-AD/CFtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python models/DAE/DAEtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python models/DFR/DFRtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python models/FAE/FAEtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python models/f-AnoGAN/GANtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python models/PADIM/PADIMtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python models/PII/PIItrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python models/RD/RDtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python models/VAE/VAEtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f
python models/VAE/r-VAEtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f

python models/AMCons/AMCtrainer.py -lp t -ev t -mod CXR 
python models/CFLOW-AD/CFtrainer.py -lp t -ev t -mod CXR  
python models/DAE/DAEtrainer.py -lp t -ev t -mod CXR  
python models/DFR/DFRtrainer.py -lp t -ev t -mod CXR  
python models/FAE/FAEtrainer.py -lp t -ev t -mod CXR  
python models/f-AnoGAN/GANtrainer.py -lp t -ev t -mod CXR  
python models/PADIM/PADIMtrainer.py -lp t -ev t -mod CXR  
python models/PII/PIItrainer.py -lp t -ev t -mod CXR  
python models/RD/RDtrainer.py -lp t -ev t -mod CXR  
python models/VAE/VAEtrainer.py -lp t -ev t -mod CXR  
python models/VAE/r-VAEtrainer.py -lp t -ev t -mod CXR

python models/AMCons/AMCtrainer.py -lp t -ev t -mod RF 
python models/CFLOW-AD/CFtrainer.py -lp t -ev t -mod RF  
python models/DAE/DAEtrainer.py -lp t -ev t -mod RF  
python models/DFR/DFRtrainer.py -lp t -ev t -mod RF  
python models/FAE/FAEtrainer.py -lp t -ev t -mod RF  
python models/f-AnoGAN/GANtrainer.py -lp t -ev t -mod RF  
python models/PADIM/PADIMtrainer.py -lp t -ev t -mod RF  
python models/PII/PIItrainer.py -lp t -ev t -mod RF  
python models/RD/RDtrainer.py -lp t -ev t -mod RF  
python models/VAE/VAEtrainer.py -lp t -ev t -mod RF  
python models/VAE/r-VAEtrainer.py -lp t -ev t -mod RF  

# seed = 20

python models/AMCons/AMCtrainer.py -lp t --seed 20 -mod MRI 
python models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -mod MRI  
python models/DAE/DAEtrainer.py -lp t --seed 20 -mod MRI  
python models/DFR/DFRtrainer.py -lp t --seed 20 -mod MRI  
python models/FAE/FAEtrainer.py -lp t --seed 20 -mod MRI  
python models/f-AnoGAN/GANtrainer.py -lp t --seed 20 -mod MRI
python models/PADIM/PADIMtrainer.py -lp t --seed 20 -mod MRI  
python models/PII/PIItrainer.py -lp t --seed 20 -mod MRI  
python models/RD/RDtrainer.py -lp t --seed 20 -mod MRI  
python models/VAE/VAEtrainer.py -lp t --seed 20 -mod MRI  

python models/AMCons/AMCtrainer.py -lp t --seed 20 -mod MRI -seq t1 
python models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -mod MRI -seq t1  
python models/DAE/DAEtrainer.py -lp t --seed 20 -mod MRI -seq t1  
python models/DFR/DFRtrainer.py -lp t --seed 20 -mod MRI -seq t1  
python models/FAE/FAEtrainer.py -lp t --seed 20 -mod MRI -seq t1  
python models/f-AnoGAN/GANtrainer.py -lp t --seed 20 -mod MRI -seq t1
python models/PADIM/PADIMtrainer.py -lp t --seed 20 -mod MRI -seq t1  
python models/PII/PIItrainer.py -lp t --seed 20 -mod MRI -seq t1  
python models/RD/RDtrainer.py -lp t --seed 20 -mod MRI -seq t1  
python models/VAE/VAEtrainer.py -lp t --seed 20 -mod MRI -seq t1

python models/AMCons/AMCtrainer.py -lp t --seed 20 -mod CXR 
python models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -mod CXR  
python models/DAE/DAEtrainer.py -lp t --seed 20 -mod CXR  
python models/DFR/DFRtrainer.py -lp t --seed 20 -mod CXR  
python models/FAE/FAEtrainer.py -lp t --seed 20 -mod CXR  
python models/f-AnoGAN/GANtrainer.py -lp t --seed 20 -mod CXR
python models/PADIM/PADIMtrainer.py -lp t --seed 20 -mod CXR  
python models/PII/PIItrainer.py -lp t --seed 20 -mod CXR  
python models/RD/RDtrainer.py -lp t --seed 20 -mod CXR  
python models/VAE/VAEtrainer.py -lp t --seed 20 -mod CXR  

python models/AMCons/AMCtrainer.py -lp t --seed 20 -mod RF 
python models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -mod RF  
python models/DAE/DAEtrainer.py -lp t --seed 20 -mod RF  
python models/DFR/DFRtrainer.py -lp t --seed 20 -mod RF  
python models/FAE/FAEtrainer.py -lp t --seed 20 -mod RF  
python models/f-AnoGAN/GANtrainer.py -lp t --seed 20 -mod RF
python models/PADIM/PADIMtrainer.py -lp t --seed 20 -mod RF  
python models/PII/PIItrainer.py -lp t --seed 20 -mod RF  
python models/RD/RDtrainer.py -lp t --seed 20 -mod RF  
python models/VAE/VAEtrainer.py -lp t --seed 20 -mod RF  
  
python models/AMCons/AMCtrainer.py -lp t --seed 20 -ev t -mod MRI 
python models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -ev t -mod MRI  
python models/DAE/DAEtrainer.py -lp t --seed 20 -ev t -mod MRI  
python models/DFR/DFRtrainer.py -lp t --seed 20 -ev t -mod MRI  
python models/FAE/FAEtrainer.py -lp t --seed 20 -ev t -mod MRI  
python models/f-AnoGAN/GANtrainer.py -lp t --seed 20 -ev t -mod MRI  
python models/PADIM/PADIMtrainer.py -lp t --seed 20 -ev t -mod MRI  
python models/PII/PIItrainer.py -lp t --seed 20 -ev t -mod MRI  
python models/RD/RDtrainer.py -lp t --seed 20 -ev t -mod MRI  
python models/VAE/VAEtrainer.py -lp t --seed 20 -ev t -mod MRI  
python models/VAE/r-VAEtrainer.py -lp t --seed 20 -ev t -mod MRI 

python models/AMCons/AMCtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 
python models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1  
python models/DAE/DAEtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1  
python models/DFR/DFRtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1  
python models/FAE/FAEtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1  
python models/f-AnoGAN/GANtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1  
python models/PADIM/PADIMtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1  
python models/PII/PIItrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1  
python models/RD/RDtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1  
python models/VAE/VAEtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1
python models/VAE/r-VAEtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1

python models/AMCons/AMCtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f 
python models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python models/DAE/DAEtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python models/DFR/DFRtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python models/FAE/FAEtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python models/f-AnoGAN/GANtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python models/PADIM/PADIMtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python models/PII/PIItrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python models/RD/RDtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python models/VAE/VAEtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f
python models/VAE/r-VAEtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f

python models/AMCons/AMCtrainer.py -lp t --seed 20 -ev t -mod CXR 
python models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -ev t -mod CXR  
python models/DAE/DAEtrainer.py -lp t --seed 20 -ev t -mod CXR  
python models/DFR/DFRtrainer.py -lp t --seed 20 -ev t -mod CXR  
python models/FAE/FAEtrainer.py -lp t --seed 20 -ev t -mod CXR  
python models/f-AnoGAN/GANtrainer.py -lp t --seed 20 -ev t -mod CXR  
python models/PADIM/PADIMtrainer.py -lp t --seed 20 -ev t -mod CXR  
python models/PII/PIItrainer.py -lp t --seed 20 -ev t -mod CXR  
python models/RD/RDtrainer.py -lp t --seed 20 -ev t -mod CXR  
python models/VAE/VAEtrainer.py -lp t --seed 20 -ev t -mod CXR  
python models/VAE/r-VAEtrainer.py -lp t --seed 20 -ev t -mod CXR

python models/AMCons/AMCtrainer.py -lp t --seed 20 -ev t -mod RF 
python models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -ev t -mod RF  
python models/DAE/DAEtrainer.py -lp t --seed 20 -ev t -mod RF  
python models/DFR/DFRtrainer.py -lp t --seed 20 -ev t -mod RF  
python models/FAE/FAEtrainer.py -lp t --seed 20 -ev t -mod RF  
python models/f-AnoGAN/GANtrainer.py -lp t --seed 20 -ev t -mod RF  
python models/PADIM/PADIMtrainer.py -lp t --seed 20 -ev t -mod RF  
python models/PII/PIItrainer.py -lp t --seed 20 -ev t -mod RF  
python models/RD/RDtrainer.py -lp t --seed 20 -ev t -mod RF  
python models/VAE/VAEtrainer.py -lp t --seed 20 -ev t -mod RF  
python models/VAE/r-VAEtrainer.py -lp t --seed 20 -ev t -mod RF  

# seed = 30
python models/AMCons/AMCtrainer.py -lp t --seed 30 -mod MRI 
python models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -mod MRI  
python models/DAE/DAEtrainer.py -lp t --seed 30 -mod MRI  
python models/DFR/DFRtrainer.py -lp t --seed 30 -mod MRI  
python models/FAE/FAEtrainer.py -lp t --seed 30 -mod MRI  
python models/f-AnoGAN/GANtrainer.py -lp t --seed 30 -mod MRI
python models/PADIM/PADIMtrainer.py -lp t --seed 30 -mod MRI  
python models/PII/PIItrainer.py -lp t --seed 30 -mod MRI  
python models/RD/RDtrainer.py -lp t --seed 30 -mod MRI  
python models/VAE/VAEtrainer.py -lp t --seed 30 -mod MRI  

python models/AMCons/AMCtrainer.py -lp t --seed 30 -mod MRI -seq t1 
python models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -mod MRI -seq t1  
python models/DAE/DAEtrainer.py -lp t --seed 30 -mod MRI -seq t1  
python models/DFR/DFRtrainer.py -lp t --seed 30 -mod MRI -seq t1  
python models/FAE/FAEtrainer.py -lp t --seed 30 -mod MRI -seq t1  
python models/f-AnoGAN/GANtrainer.py -lp t --seed 30 -mod MRI -seq t1
python models/PADIM/PADIMtrainer.py -lp t --seed 30 -mod MRI -seq t1  
python models/PII/PIItrainer.py -lp t --seed 30 -mod MRI -seq t1  
python models/RD/RDtrainer.py -lp t --seed 30 -mod MRI -seq t1  
python models/VAE/VAEtrainer.py -lp t --seed 30 -mod MRI -seq t1

python models/AMCons/AMCtrainer.py -lp t --seed 30 -mod CXR 
python models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -mod CXR  
python models/DAE/DAEtrainer.py -lp t --seed 30 -mod CXR  
python models/DFR/DFRtrainer.py -lp t --seed 30 -mod CXR  
python models/FAE/FAEtrainer.py -lp t --seed 30 -mod CXR  
python models/f-AnoGAN/GANtrainer.py -lp t --seed 30 -mod CXR
python models/PADIM/PADIMtrainer.py -lp t --seed 30 -mod CXR  
python models/PII/PIItrainer.py -lp t --seed 30 -mod CXR  
python models/RD/RDtrainer.py -lp t --seed 30 -mod CXR  
python models/VAE/VAEtrainer.py -lp t --seed 30 -mod CXR  

python models/AMCons/AMCtrainer.py -lp t --seed 30 -mod RF 
python models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -mod RF  
python models/DAE/DAEtrainer.py -lp t --seed 30 -mod RF  
python models/DFR/DFRtrainer.py -lp t --seed 30 -mod RF  
python models/FAE/FAEtrainer.py -lp t --seed 30 -mod RF  
python models/f-AnoGAN/GANtrainer.py -lp t --seed 30 -mod RF
python models/PADIM/PADIMtrainer.py -lp t --seed 30 -mod RF  
python models/PII/PIItrainer.py -lp t --seed 30 -mod RF  
python models/RD/RDtrainer.py -lp t --seed 30 -mod RF  
python models/VAE/VAEtrainer.py -lp t --seed 30 -mod RF  
  
python models/AMCons/AMCtrainer.py -lp t --seed 30 -ev t -mod MRI 
python models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -ev t -mod MRI  
python models/DAE/DAEtrainer.py -lp t --seed 30 -ev t -mod MRI  
python models/DFR/DFRtrainer.py -lp t --seed 30 -ev t -mod MRI  
python models/FAE/FAEtrainer.py -lp t --seed 30 -ev t -mod MRI  
python models/f-AnoGAN/GANtrainer.py -lp t --seed 30 -ev t -mod MRI  
python models/PADIM/PADIMtrainer.py -lp t --seed 30 -ev t -mod MRI  
python models/PII/PIItrainer.py -lp t --seed 30 -ev t -mod MRI  
python models/RD/RDtrainer.py -lp t --seed 30 -ev t -mod MRI  
python models/VAE/VAEtrainer.py -lp t --seed 30 -ev t -mod MRI  
python models/VAE/r-VAEtrainer.py -lp t --seed 30 -ev t -mod MRI 

python models/AMCons/AMCtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 
python models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1  
python models/DAE/DAEtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1  
python models/DFR/DFRtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1  
python models/FAE/FAEtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1  
python models/f-AnoGAN/GANtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1  
python models/PADIM/PADIMtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1  
python models/PII/PIItrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1  
python models/RD/RDtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1  
python models/VAE/VAEtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1
python models/VAE/r-VAEtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1

python models/AMCons/AMCtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f 
python models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python models/DAE/DAEtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python models/DFR/DFRtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python models/FAE/FAEtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python models/f-AnoGAN/GANtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python models/PADIM/PADIMtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python models/PII/PIItrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python models/RD/RDtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python models/VAE/VAEtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f
python models/VAE/r-VAEtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f

python models/AMCons/AMCtrainer.py -lp t --seed 30 -ev t -mod CXR 
python models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -ev t -mod CXR  
python models/DAE/DAEtrainer.py -lp t --seed 30 -ev t -mod CXR  
python models/DFR/DFRtrainer.py -lp t --seed 30 -ev t -mod CXR  
python models/FAE/FAEtrainer.py -lp t --seed 30 -ev t -mod CXR  
python models/f-AnoGAN/GANtrainer.py -lp t --seed 30 -ev t -mod CXR  
python models/PADIM/PADIMtrainer.py -lp t --seed 30 -ev t -mod CXR  
python models/PII/PIItrainer.py -lp t --seed 30 -ev t -mod CXR  
python models/RD/RDtrainer.py -lp t --seed 30 -ev t -mod CXR  
python models/VAE/VAEtrainer.py -lp t --seed 30 -ev t -mod CXR  
python models/VAE/r-VAEtrainer.py -lp t --seed 30 -ev t -mod CXR

python models/AMCons/AMCtrainer.py -lp t --seed 30 -ev t -mod RF 
python models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -ev t -mod RF  
python models/DAE/DAEtrainer.py -lp t --seed 30 -ev t -mod RF  
python models/DFR/DFRtrainer.py -lp t --seed 30 -ev t -mod RF  
python models/FAE/FAEtrainer.py -lp t --seed 30 -ev t -mod RF  
python models/f-AnoGAN/GANtrainer.py -lp t --seed 30 -ev t -mod RF  
python models/PADIM/PADIMtrainer.py -lp t --seed 30 -ev t -mod RF  
python models/PII/PIItrainer.py -lp t --seed 30 -ev t -mod RF  
python models/RD/RDtrainer.py -lp t --seed 30 -ev t -mod RF  
python models/VAE/VAEtrainer.py -lp t --seed 30 -ev t -mod RF  
python models/VAE/r-VAEtrainer.py -lp t --seed 30 -ev t -mod RF  

