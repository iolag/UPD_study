# Activate the virtual environment
# source /opt/anaconda3/etc/profile.d/conda.sh
# conda activate upd

#Method training

python UPD_study/models/AMCons/AMCtrainer.py --seed 20 -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 20 -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py --seed 20 -mod MRI  
#python UPD_study/models/DFR/DFRtrainer.py --seed 20 -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py --seed 20 -mod MRI  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 20 -mod MRI
python UPD_study/models/PADIM/PADIMtrainer.py --seed 20 -mod MRI  
python UPD_study/models/PII/PIItrainer.py --seed 20 -mod MRI  
python UPD_study/models/RD/RDtrainer.py --seed 20 -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py --seed 20 -mod MRI  

python UPD_study/models/AMCons/AMCtrainer.py --seed 20 -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 20 -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py --seed 20 -mod MRI -seq t1  
#python UPD_study/models/DFR/DFRtrainer.py --seed 20 -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py --seed 20 -mod MRI -seq t1  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 20 -mod MRI -seq t1
python UPD_study/models/PADIM/PADIMtrainer.py --seed 20 -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py --seed 20 -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py --seed 20 -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py --seed 20 -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py --seed 20 -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 20 -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py --seed 20 -mod CXR  
#python UPD_study/models/DFR/DFRtrainer.py --seed 20 -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py --seed 20 -mod CXR  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 20 -mod CXR
python UPD_study/models/PADIM/PADIMtrainer.py --seed 20 -mod CXR  
python UPD_study/models/PII/PIItrainer.py --seed 20 -mod CXR  
python UPD_study/models/RD/RDtrainer.py --seed 20 -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py --seed 20 -mod CXR  

python UPD_study/models/AMCons/AMCtrainer.py --seed 20 -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 20 -mod RF  
python UPD_study/models/DAE/DAEtrainer.py --seed 20 -mod RF  
#python UPD_study/models/DFR/DFRtrainer.py --seed 20 -mod RF  
python UPD_study/models/FAE/FAEtrainer.py --seed 20 -mod RF  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 20 -mod RF
python UPD_study/models/PADIM/PADIMtrainer.py --seed 20 -mod RF  
python UPD_study/models/PII/PIItrainer.py --seed 20 -mod RF  
python UPD_study/models/RD/RDtrainer.py --seed 20 -mod RF  
python UPD_study/models/VAE/VAEtrainer.py --seed 20 -mod RF  

#Method evaluation

python UPD_study/models/AMCons/AMCtrainer.py --seed 20 -ev t -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 20 -ev t -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py --seed 20 -ev t -mod MRI  
#python UPD_study/models/DFR/DFRtrainer.py --seed 20 -ev t -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py --seed 20 -ev t -mod MRI  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 20 -ev t -mod MRI  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 20 -ev t -mod MRI  
python UPD_study/models/PII/PIItrainer.py --seed 20 -ev t -mod MRI  
python UPD_study/models/RD/RDtrainer.py --seed 20 -ev t -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py --seed 20 -ev t -mod MRI  
python UPD_study/models/VAE/r-VAEtrainer.py --seed 20 -ev t -mod MRI 

python UPD_study/models/AMCons/AMCtrainer.py --seed 20 -ev t -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py --seed 20 -ev t -mod MRI -seq t1  
#python UPD_study/models/DFR/DFRtrainer.py --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py --seed 20 -ev t -mod MRI -seq t1
python UPD_study/models/VAE/r-VAEtrainer.py --seed 20 -ev t -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/DAE/DAEtrainer.py --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
#python UPD_study/models/DFR/DFRtrainer.py --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/FAE/FAEtrainer.py --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/PII/PIItrainer.py --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/RD/RDtrainer.py --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/VAE/VAEtrainer.py --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f
python UPD_study/models/VAE/r-VAEtrainer.py --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f

python UPD_study/models/AMCons/AMCtrainer.py --seed 20 -ev t -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 20 -ev t -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py --seed 20 -ev t -mod CXR  
#python UPD_study/models/DFR/DFRtrainer.py --seed 20 -ev t -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py --seed 20 -ev t -mod CXR  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 20 -ev t -mod CXR  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 20 -ev t -mod CXR  
python UPD_study/models/PII/PIItrainer.py --seed 20 -ev t -mod CXR  
python UPD_study/models/RD/RDtrainer.py --seed 20 -ev t -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py --seed 20 -ev t -mod CXR  
python UPD_study/models/VAE/r-VAEtrainer.py --seed 20 -ev t -mod CXR

python UPD_study/models/AMCons/AMCtrainer.py --seed 20 -ev t -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 20 -ev t -mod RF  
python UPD_study/models/DAE/DAEtrainer.py --seed 20 -ev t -mod RF  
#python UPD_study/models/DFR/DFRtrainer.py --seed 20 -ev t -mod RF  
python UPD_study/models/FAE/FAEtrainer.py --seed 20 -ev t -mod RF  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 20 -ev t -mod RF  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 20 -ev t -mod RF  
python UPD_study/models/PII/PIItrainer.py --seed 20 -ev t -mod RF  
python UPD_study/models/RD/RDtrainer.py --seed 20 -ev t -mod RF  
python UPD_study/models/VAE/VAEtrainer.py --seed 20 -ev t -mod RF  
python UPD_study/models/VAE/r-VAEtrainer.py --seed 20 -ev t -mod RF  


python UPD_study/models/AMCons/AMCtrainer.py --seed 10 -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 10 -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py --seed 10 -mod MRI  
#python UPD_study/models/DFR/DFRtrainer.py --seed 10 -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py --seed 10 -mod MRI  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 10 -mod MRI
python UPD_study/models/PADIM/PADIMtrainer.py --seed 10 -mod MRI  
python UPD_study/models/PII/PIItrainer.py --seed 10 -mod MRI  
python UPD_study/models/RD/RDtrainer.py --seed 10 -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py --seed 10 -mod MRI  

python UPD_study/models/AMCons/AMCtrainer.py --seed 10 -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 10 -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py --seed 10 -mod MRI -seq t1  
#python UPD_study/models/DFR/DFRtrainer.py --seed 10 -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py --seed 10 -mod MRI -seq t1  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 10 -mod MRI -seq t1
python UPD_study/models/PADIM/PADIMtrainer.py --seed 10 -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py --seed 10 -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py --seed 10 -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py --seed 10 -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py --seed 10 -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 10 -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py --seed 10 -mod CXR  
#python UPD_study/models/DFR/DFRtrainer.py --seed 10 -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py --seed 10 -mod CXR  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 10 -mod CXR
python UPD_study/models/PADIM/PADIMtrainer.py --seed 10 -mod CXR  
python UPD_study/models/PII/PIItrainer.py --seed 10 -mod CXR  
python UPD_study/models/RD/RDtrainer.py --seed 10 -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py --seed 10 -mod CXR  

python UPD_study/models/AMCons/AMCtrainer.py --seed 10 -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 10 -mod RF  
python UPD_study/models/DAE/DAEtrainer.py --seed 10 -mod RF  
#python UPD_study/models/DFR/DFRtrainer.py --seed 10 -mod RF  
python UPD_study/models/FAE/FAEtrainer.py --seed 10 -mod RF  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 10 -mod RF
python UPD_study/models/PADIM/PADIMtrainer.py --seed 10 -mod RF  
python UPD_study/models/PII/PIItrainer.py --seed 10 -mod RF  
python UPD_study/models/RD/RDtrainer.py --seed 10 -mod RF  
python UPD_study/models/VAE/VAEtrainer.py --seed 10 -mod RF  

#Method evaluation

python UPD_study/models/AMCons/AMCtrainer.py --seed 10 -ev t -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 10 -ev t -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py --seed 10 -ev t -mod MRI  
#python UPD_study/models/DFR/DFRtrainer.py --seed 10 -ev t -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py --seed 10 -ev t -mod MRI  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 10 -ev t -mod MRI  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 10 -ev t -mod MRI  
python UPD_study/models/PII/PIItrainer.py --seed 10 -ev t -mod MRI  
python UPD_study/models/RD/RDtrainer.py --seed 10 -ev t -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py --seed 10 -ev t -mod MRI  
python UPD_study/models/VAE/r-VAEtrainer.py --seed 10 -ev t -mod MRI 

python UPD_study/models/AMCons/AMCtrainer.py --seed 10 -ev t -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 10 -ev t -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py --seed 10 -ev t -mod MRI -seq t1  
#python UPD_study/models/DFR/DFRtrainer.py --seed 10 -ev t -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py --seed 10 -ev t -mod MRI -seq t1  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 10 -ev t -mod MRI -seq t1  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 10 -ev t -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py --seed 10 -ev t -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py --seed 10 -ev t -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py --seed 10 -ev t -mod MRI -seq t1
python UPD_study/models/VAE/r-VAEtrainer.py --seed 10 -ev t -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py --seed 10 -ev t -mod MRI -seq t1 --brats_t1 f 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 10 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/DAE/DAEtrainer.py --seed 10 -ev t -mod MRI -seq t1 --brats_t1 f  
#python UPD_study/models/DFR/DFRtrainer.py --seed 10 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/FAE/FAEtrainer.py --seed 10 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 10 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 10 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/PII/PIItrainer.py --seed 10 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/RD/RDtrainer.py --seed 10 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/VAE/VAEtrainer.py --seed 10 -ev t -mod MRI -seq t1 --brats_t1 f
python UPD_study/models/VAE/r-VAEtrainer.py --seed 10 -ev t -mod MRI -seq t1 --brats_t1 f

python UPD_study/models/AMCons/AMCtrainer.py --seed 10 -ev t -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 10 -ev t -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py --seed 10 -ev t -mod CXR  
#python UPD_study/models/DFR/DFRtrainer.py --seed 10 -ev t -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py --seed 10 -ev t -mod CXR  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 10 -ev t -mod CXR  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 10 -ev t -mod CXR  
python UPD_study/models/PII/PIItrainer.py --seed 10 -ev t -mod CXR  
python UPD_study/models/RD/RDtrainer.py --seed 10 -ev t -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py --seed 10 -ev t -mod CXR  
python UPD_study/models/VAE/r-VAEtrainer.py --seed 10 -ev t -mod CXR

python UPD_study/models/AMCons/AMCtrainer.py --seed 10 -ev t -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 10 -ev t -mod RF  
python UPD_study/models/DAE/DAEtrainer.py --seed 10 -ev t -mod RF  
#python UPD_study/models/DFR/DFRtrainer.py --seed 10 -ev t -mod RF  
python UPD_study/models/FAE/FAEtrainer.py --seed 10 -ev t -mod RF  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 10 -ev t -mod RF  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 10 -ev t -mod RF  
python UPD_study/models/PII/PIItrainer.py --seed 10 -ev t -mod RF  
python UPD_study/models/RD/RDtrainer.py --seed 10 -ev t -mod RF  
python UPD_study/models/VAE/VAEtrainer.py --seed 10 -ev t -mod RF  
python UPD_study/models/VAE/r-VAEtrainer.py --seed 10 -ev t -mod RF  


python UPD_study/models/AMCons/AMCtrainer.py --seed 30 -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 30 -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py --seed 30 -mod MRI  
#python UPD_study/models/DFR/DFRtrainer.py --seed 30 -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py --seed 30 -mod MRI  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 30 -mod MRI
python UPD_study/models/PADIM/PADIMtrainer.py --seed 30 -mod MRI  
python UPD_study/models/PII/PIItrainer.py --seed 30 -mod MRI  
python UPD_study/models/RD/RDtrainer.py --seed 30 -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py --seed 30 -mod MRI  

python UPD_study/models/AMCons/AMCtrainer.py --seed 30 -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 30 -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py --seed 30 -mod MRI -seq t1  
#python UPD_study/models/DFR/DFRtrainer.py --seed 30 -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py --seed 30 -mod MRI -seq t1  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 30 -mod MRI -seq t1
python UPD_study/models/PADIM/PADIMtrainer.py --seed 30 -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py --seed 30 -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py --seed 30 -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py --seed 30 -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py --seed 30 -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 30 -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py --seed 30 -mod CXR  
#python UPD_study/models/DFR/DFRtrainer.py --seed 30 -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py --seed 30 -mod CXR  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 30 -mod CXR
python UPD_study/models/PADIM/PADIMtrainer.py --seed 30 -mod CXR  
python UPD_study/models/PII/PIItrainer.py --seed 30 -mod CXR  
python UPD_study/models/RD/RDtrainer.py --seed 30 -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py --seed 30 -mod CXR  

python UPD_study/models/AMCons/AMCtrainer.py --seed 30 -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 30 -mod RF  
python UPD_study/models/DAE/DAEtrainer.py --seed 30 -mod RF  
#python UPD_study/models/DFR/DFRtrainer.py --seed 30 -mod RF  
python UPD_study/models/FAE/FAEtrainer.py --seed 30 -mod RF  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 30 -mod RF
python UPD_study/models/PADIM/PADIMtrainer.py --seed 30 -mod RF  
python UPD_study/models/PII/PIItrainer.py --seed 30 -mod RF  
python UPD_study/models/RD/RDtrainer.py --seed 30 -mod RF  
python UPD_study/models/VAE/VAEtrainer.py --seed 30 -mod RF  

#Method evaluation

python UPD_study/models/AMCons/AMCtrainer.py --seed 30 -ev t -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 30 -ev t -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py --seed 30 -ev t -mod MRI  
#python UPD_study/models/DFR/DFRtrainer.py --seed 30 -ev t -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py --seed 30 -ev t -mod MRI  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 30 -ev t -mod MRI  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 30 -ev t -mod MRI  
python UPD_study/models/PII/PIItrainer.py --seed 30 -ev t -mod MRI  
python UPD_study/models/RD/RDtrainer.py --seed 30 -ev t -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py --seed 30 -ev t -mod MRI  
python UPD_study/models/VAE/r-VAEtrainer.py --seed 30 -ev t -mod MRI 

python UPD_study/models/AMCons/AMCtrainer.py --seed 30 -ev t -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py --seed 30 -ev t -mod MRI -seq t1  
#python UPD_study/models/DFR/DFRtrainer.py --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py --seed 30 -ev t -mod MRI -seq t1
python UPD_study/models/VAE/r-VAEtrainer.py --seed 30 -ev t -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/DAE/DAEtrainer.py --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
#python UPD_study/models/DFR/DFRtrainer.py --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/FAE/FAEtrainer.py --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/PII/PIItrainer.py --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/RD/RDtrainer.py --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/VAE/VAEtrainer.py --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f
python UPD_study/models/VAE/r-VAEtrainer.py --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f

python UPD_study/models/AMCons/AMCtrainer.py --seed 30 -ev t -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 30 -ev t -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py --seed 30 -ev t -mod CXR  
#python UPD_study/models/DFR/DFRtrainer.py --seed 30 -ev t -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py --seed 30 -ev t -mod CXR  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 30 -ev t -mod CXR  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 30 -ev t -mod CXR  
python UPD_study/models/PII/PIItrainer.py --seed 30 -ev t -mod CXR  
python UPD_study/models/RD/RDtrainer.py --seed 30 -ev t -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py --seed 30 -ev t -mod CXR  
python UPD_study/models/VAE/r-VAEtrainer.py --seed 30 -ev t -mod CXR

python UPD_study/models/AMCons/AMCtrainer.py --seed 30 -ev t -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py --seed 30 -ev t -mod RF  
python UPD_study/models/DAE/DAEtrainer.py --seed 30 -ev t -mod RF  
#python UPD_study/models/DFR/DFRtrainer.py --seed 30 -ev t -mod RF  
python UPD_study/models/FAE/FAEtrainer.py --seed 30 -ev t -mod RF  
python UPD_study/models/f-AnoGAN/GANtrainer.py --seed 30 -ev t -mod RF  
python UPD_study/models/PADIM/PADIMtrainer.py --seed 30 -ev t -mod RF  
python UPD_study/models/PII/PIItrainer.py --seed 30 -ev t -mod RF  
python UPD_study/models/RD/RDtrainer.py --seed 30 -ev t -mod RF  
python UPD_study/models/VAE/VAEtrainer.py --seed 30 -ev t -mod RF  
python UPD_study/models/VAE/r-VAEtrainer.py --seed 30 -ev t -mod RF  

