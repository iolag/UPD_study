# # Activate the virtual environment
# source /opt/anaconda3/etc/profile.d/conda.sh
# conda activate upd
# Pre-training backbones with CCD 

#seed 10
python UPD_study/models/CCD/CCDtrainer.py -arch resnet18 -mod MRI 
python UPD_study/models/CCD/CCDtrainer.py -arch wide_resnet50_2 -mod MRI 
python UPD_study/models/CCD/CCDtrainer.py -arch vae -mod MRI 
python UPD_study/models/CCD/CCDtrainer.py -arch fanogan -mod MRI 
python UPD_study/models/CCD/CCDtrainer.py -arch unet -mod MRI 
python UPD_study/models/CCD/CCDtrainer.py -arch pii -mod MRI 
python UPD_study/models/CCD/CCDtrainer.py -arch amc -mod MRI 

python UPD_study/models/CCD/CCDtrainer.py -arch resnet18 -mod MRI -seq t1 
python UPD_study/models/CCD/CCDtrainer.py -arch wide_resnet50_2 -mod MRI -seq t1 
python UPD_study/models/CCD/CCDtrainer.py -arch vae -mod MRI -seq t1 
python UPD_study/models/CCD/CCDtrainer.py -arch fanogan -mod MRI -seq t1 
python UPD_study/models/CCD/CCDtrainer.py -arch unet -mod MRI -seq t1 
python UPD_study/models/CCD/CCDtrainer.py -arch pii -mod MRI -seq t1 
python UPD_study/models/CCD/CCDtrainer.py -arch amc -mod MRI -seq t1

python UPD_study/models/CCD/CCDtrainer.py -arch resnet18 -mod CXR 
python UPD_study/models/CCD/CCDtrainer.py -arch wide_resnet50_2 -mod CXR 
python UPD_study/models/CCD/CCDtrainer.py -arch vae -mod CXR 
python UPD_study/models/CCD/CCDtrainer.py -arch fanogan -mod CXR 
python UPD_study/models/CCD/CCDtrainer.py -arch unet -mod CXR 
python UPD_study/models/CCD/CCDtrainer.py -arch pii -mod CXR 
python UPD_study/models/CCD/CCDtrainer.py -arch amc -mod CXR

python UPD_study/models/CCD/CCDtrainer.py -arch resnet18 -mod RF 
python UPD_study/models/CCD/CCDtrainer.py -arch wide_resnet50_2 -mod RF 
python UPD_study/models/CCD/CCDtrainer.py -arch vae -mod RF 
python UPD_study/models/CCD/CCDtrainer.py -arch fanogan -mod RF 
python UPD_study/models/CCD/CCDtrainer.py -arch unet -mod RF 
python UPD_study/models/CCD/CCDtrainer.py -arch pii -mod RF 
python UPD_study/models/CCD/CCDtrainer.py -arch amc -mod RF

python UPD_study/models/CCD/CCDtrainer.py -arch expvae
python UPD_study/models/CCD/CCDtrainer.py -arch expvae -seq t1
python UPD_study/models/CCD/CCDtrainer.py -arch expvae -mod RF
python UPD_study/models/CCD/CCDtrainer.py -arch expvae -mod CXR

python UPD_study/models/CCD/CCDtrainer.py -arch resnet18
python UPD_study/models/CCD/CCDtrainer.py -arch resnet18 -seq t1
python UPD_study/models/CCD/CCDtrainer.py -arch resnet18 -mod RF
python UPD_study/models/CCD/CCDtrainer.py -arch resnet18 -mod CXR

#seed 20
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch resnet18 -mod MRI 
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch wide_resnet50_2 -mod MRI 
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch vae -mod MRI 
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch fanogan -mod MRI 
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch unet -mod MRI 
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch pii -mod MRI 
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch amc -mod MRI 

python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch resnet18 -mod MRI -seq t1 
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch wide_resnet50_2 -mod MRI -seq t1 
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch vae -mod MRI -seq t1 
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch fanogan -mod MRI -seq t1 
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch unet -mod MRI -seq t1 
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch pii -mod MRI -seq t1 
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch amc -mod MRI -seq t1

python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch resnet18 -mod CXR 
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch wide_resnet50_2 -mod CXR 
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch vae -mod CXR 
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch fanogan -mod CXR 
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch unet -mod CXR 
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch pii -mod CXR 
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch amc -mod CXR

python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch resnet18 -mod RF 
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch wide_resnet50_2 -mod RF 
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch vae -mod RF 
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch fanogan -mod RF 
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch unet -mod RF 
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch pii -mod RF 
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch amc -mod RF

python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch expvae
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch expvae -seq t1
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch expvae -mod RF
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch expvae -mod CXR

python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch resnet18
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch resnet18 -seq t1
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch resnet18 -mod RF
python UPD_study/models/CCD/CCDtrainer.py --seed 20 -arch resnet18 -mod CXR

#seed 30
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch resnet18 -mod MRI 
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch wide_resnet50_2 -mod MRI 
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch vae -mod MRI 
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch fanogan -mod MRI 
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch unet -mod MRI 
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch pii -mod MRI 
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch amc -mod MRI 

python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch resnet18 -mod MRI -seq t1 
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch wide_resnet50_2 -mod MRI -seq t1 
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch vae -mod MRI -seq t1 
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch fanogan -mod MRI -seq t1 
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch unet -mod MRI -seq t1 
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch pii -mod MRI -seq t1 
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch amc -mod MRI -seq t1

python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch resnet18 -mod CXR 
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch wide_resnet50_2 -mod CXR 
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch vae -mod CXR 
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch fanogan -mod CXR 
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch unet -mod CXR 
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch pii -mod CXR 
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch amc -mod CXR

python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch resnet18 -mod RF 
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch wide_resnet50_2 -mod RF 
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch vae -mod RF 
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch fanogan -mod RF 
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch unet -mod RF 
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch pii -mod RF 
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch amc -mod RF

python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch expvae
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch expvae -seq t1
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch expvae -mod RF
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch expvae -mod CXR

python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch resnet18
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch resnet18 -seq t1
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch resnet18 -mod RF
python UPD_study/models/CCD/CCDtrainer.py --seed 30 -arch resnet18 -mod CXR


#Method training
# seed = 10

python UPD_study/models/AMCons/AMCtrainer.py -lp t -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py -lp t -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py -lp t -mod MRI  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t -mod MRI
python UPD_study/models/PADIM/PADIMtrainer.py -lp t -mod MRI  
python UPD_study/models/PII/PIItrainer.py -lp t -mod MRI  
python UPD_study/models/RD/RDtrainer.py -lp t -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py -lp t -mod MRI  

python UPD_study/models/AMCons/AMCtrainer.py -lp t -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py -lp t -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py -lp t -mod MRI -seq t1  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t -mod MRI -seq t1
python UPD_study/models/PADIM/PADIMtrainer.py -lp t -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py -lp t -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py -lp t -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py -lp t -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py -lp t -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py -lp t -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py -lp t -mod CXR  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t -mod CXR
python UPD_study/models/PADIM/PADIMtrainer.py -lp t -mod CXR  
python UPD_study/models/PII/PIItrainer.py -lp t -mod CXR  
python UPD_study/models/RD/RDtrainer.py -lp t -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py -lp t -mod CXR  

python UPD_study/models/AMCons/AMCtrainer.py -lp t -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t -mod RF  
python UPD_study/models/DAE/DAEtrainer.py -lp t -mod RF  
python UPD_study/models/FAE/FAEtrainer.py -lp t -mod RF  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t -mod RF
python UPD_study/models/PADIM/PADIMtrainer.py -lp t -mod RF  
python UPD_study/models/PII/PIItrainer.py -lp t -mod RF  
python UPD_study/models/RD/RDtrainer.py -lp t -mod RF  
python UPD_study/models/VAE/VAEtrainer.py -lp t -mod RF  
  
python UPD_study/models/AMCons/AMCtrainer.py -lp t -ev t -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t -ev t -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py -lp t -ev t -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py -lp t -ev t -mod MRI  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t -ev t -mod MRI  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t -ev t -mod MRI  
python UPD_study/models/PII/PIItrainer.py -lp t -ev t -mod MRI  
python UPD_study/models/RD/RDtrainer.py -lp t -ev t -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py -lp t -ev t -mod MRI  
python UPD_study/models/VAE/r-VAEtrainer.py -lp t -ev t -mod MRI 

python UPD_study/models/AMCons/AMCtrainer.py -lp t -ev t -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t -ev t -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py -lp t -ev t -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py -lp t -ev t -mod MRI -seq t1  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t -ev t -mod MRI -seq t1  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t -ev t -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py -lp t -ev t -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py -lp t -ev t -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py -lp t -ev t -mod MRI -seq t1
python UPD_study/models/VAE/r-VAEtrainer.py -lp t -ev t -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/DAE/DAEtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/FAE/FAEtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/PII/PIItrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/RD/RDtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/VAE/VAEtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f
python UPD_study/models/VAE/r-VAEtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f

python UPD_study/models/AMCons/AMCtrainer.py -lp t -ev t -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t -ev t -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py -lp t -ev t -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py -lp t -ev t -mod CXR  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t -ev t -mod CXR  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t -ev t -mod CXR  
python UPD_study/models/PII/PIItrainer.py -lp t -ev t -mod CXR  
python UPD_study/models/RD/RDtrainer.py -lp t -ev t -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py -lp t -ev t -mod CXR  
python UPD_study/models/VAE/r-VAEtrainer.py -lp t -ev t -mod CXR

python UPD_study/models/AMCons/AMCtrainer.py -lp t -ev t -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t -ev t -mod RF  
python UPD_study/models/DAE/DAEtrainer.py -lp t -ev t -mod RF  
python UPD_study/models/FAE/FAEtrainer.py -lp t -ev t -mod RF  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t -ev t -mod RF  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t -ev t -mod RF  
python UPD_study/models/PII/PIItrainer.py -lp t -ev t -mod RF  
python UPD_study/models/RD/RDtrainer.py -lp t -ev t -mod RF  
python UPD_study/models/VAE/VAEtrainer.py -lp t -ev t -mod RF  
python UPD_study/models/VAE/r-VAEtrainer.py -lp t -ev t -mod RF  

# seed = 20

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 20 -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 20 -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 20 -mod MRI  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t --seed 20 -mod MRI
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 20 -mod MRI  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 20 -mod MRI  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 20 -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 20 -mod MRI  

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 20 -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 20 -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 20 -mod MRI -seq t1  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t --seed 20 -mod MRI -seq t1
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 20 -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 20 -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 20 -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 20 -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 20 -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 20 -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 20 -mod CXR  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t --seed 20 -mod CXR
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 20 -mod CXR  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 20 -mod CXR  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 20 -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 20 -mod CXR  

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 20 -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -mod RF  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 20 -mod RF  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 20 -mod RF  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t --seed 20 -mod RF
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 20 -mod RF  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 20 -mod RF  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 20 -mod RF  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 20 -mod RF  
  
python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 20 -ev t -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -ev t -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 20 -ev t -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 20 -ev t -mod MRI  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t --seed 20 -ev t -mod MRI  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 20 -ev t -mod MRI  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 20 -ev t -mod MRI  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 20 -ev t -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 20 -ev t -mod MRI  
python UPD_study/models/VAE/r-VAEtrainer.py -lp t --seed 20 -ev t -mod MRI 

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1
python UPD_study/models/VAE/r-VAEtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f
python UPD_study/models/VAE/r-VAEtrainer.py -lp t --seed 20 -ev t -mod MRI -seq t1 --brats_t1 f

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 20 -ev t -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -ev t -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 20 -ev t -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 20 -ev t -mod CXR  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t --seed 20 -ev t -mod CXR  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 20 -ev t -mod CXR  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 20 -ev t -mod CXR  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 20 -ev t -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 20 -ev t -mod CXR  
python UPD_study/models/VAE/r-VAEtrainer.py -lp t --seed 20 -ev t -mod CXR

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 20 -ev t -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 20 -ev t -mod RF  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 20 -ev t -mod RF  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 20 -ev t -mod RF  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t --seed 20 -ev t -mod RF  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 20 -ev t -mod RF  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 20 -ev t -mod RF  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 20 -ev t -mod RF  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 20 -ev t -mod RF  
python UPD_study/models/VAE/r-VAEtrainer.py -lp t --seed 20 -ev t -mod RF  

# seed = 30
python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 30 -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 30 -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 30 -mod MRI  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t --seed 30 -mod MRI
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 30 -mod MRI  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 30 -mod MRI  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 30 -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 30 -mod MRI  

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 30 -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 30 -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 30 -mod MRI -seq t1  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t --seed 30 -mod MRI -seq t1
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 30 -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 30 -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 30 -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 30 -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 30 -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 30 -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 30 -mod CXR  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t --seed 30 -mod CXR
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 30 -mod CXR  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 30 -mod CXR  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 30 -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 30 -mod CXR  

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 30 -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -mod RF  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 30 -mod RF  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 30 -mod RF  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t --seed 30 -mod RF
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 30 -mod RF  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 30 -mod RF  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 30 -mod RF  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 30 -mod RF  
  
python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 30 -ev t -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -ev t -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 30 -ev t -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 30 -ev t -mod MRI  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t --seed 30 -ev t -mod MRI  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 30 -ev t -mod MRI  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 30 -ev t -mod MRI  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 30 -ev t -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 30 -ev t -mod MRI  
python UPD_study/models/VAE/r-VAEtrainer.py -lp t --seed 30 -ev t -mod MRI 

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1
python UPD_study/models/VAE/r-VAEtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f
python UPD_study/models/VAE/r-VAEtrainer.py -lp t --seed 30 -ev t -mod MRI -seq t1 --brats_t1 f

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 30 -ev t -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -ev t -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 30 -ev t -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 30 -ev t -mod CXR  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t --seed 30 -ev t -mod CXR  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 30 -ev t -mod CXR  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 30 -ev t -mod CXR  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 30 -ev t -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 30 -ev t -mod CXR  
python UPD_study/models/VAE/r-VAEtrainer.py -lp t --seed 30 -ev t -mod CXR

python UPD_study/models/AMCons/AMCtrainer.py -lp t --seed 30 -ev t -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t --seed 30 -ev t -mod RF  
python UPD_study/models/DAE/DAEtrainer.py -lp t --seed 30 -ev t -mod RF  
python UPD_study/models/FAE/FAEtrainer.py -lp t --seed 30 -ev t -mod RF  
python UPD_study/models/fAnoGAN/GANtrainer.py -lp t --seed 30 -ev t -mod RF  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t --seed 30 -ev t -mod RF  
python UPD_study/models/PII/PIItrainer.py -lp t --seed 30 -ev t -mod RF  
python UPD_study/models/RD/RDtrainer.py -lp t --seed 30 -ev t -mod RF  
python UPD_study/models/VAE/VAEtrainer.py -lp t --seed 30 -ev t -mod RF  
python UPD_study/models/VAE/r-VAEtrainer.py -lp t --seed 30 -ev t -mod RF  


#newly added methods: expVAE, CutPaste

#expVAE

python UPD_study/models/expVAE/expVAEtrainer.py -lp t  -mod MRI --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t  -mod MRI --seed 20 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t  -mod MRI --seed 30
python UPD_study/models/expVAE/expVAEtrainer.py -lp t  -mod MRI -seq t1 --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t  -mod MRI -seq t1 --seed 20 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t  -mod MRI -seq t1 --seed 30
python UPD_study/models/expVAE/expVAEtrainer.py -lp t  -mod MRI --brats_t1 f -seq t1 --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t  -mod MRI --brats_t1 f -seq t1 --seed 20 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t  -mod MRI --brats_t1 f -seq t1 --seed 30
python UPD_study/models/expVAE/expVAEtrainer.py -lp t  -mod RF --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t  -mod RF --seed 20 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t  -mod RF --seed 30
python UPD_study/models/expVAE/expVAEtrainer.py -lp t  -mod CXR --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t  -mod CXR --seed 20 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t  -mod CXR --seed 30

python UPD_study/models/expVAE/expVAEtrainer.py -lp t -ev t -mod MRI --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t -ev t  -mod MRI --seed 20 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t -ev t  -mod MRI --seed 30
python UPD_study/models/expVAE/expVAEtrainer.py -lp t -ev t  -mod MRI -seq t1 --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t -ev t  -mod MRI -seq t1 --seed 20 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t -ev t  -mod MRI -seq t1 --seed 30
python UPD_study/models/expVAE/expVAEtrainer.py -lp t -ev t  -mod MRI --brats_t1 f -seq t1 --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t -ev t  -mod MRI --brats_t1 f -seq t1 --seed 20 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t -ev t  -mod MRI --brats_t1 f -seq t1 --seed 30
python UPD_study/models/expVAE/expVAEtrainer.py -lp t -ev t  -mod RF --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t -ev t  -mod RF --seed 20 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t -ev t  -mod RF --seed 30
python UPD_study/models/expVAE/expVAEtrainer.py -lp t -ev t  -mod CXR --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t -ev t  -mod CXR --seed 20 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t -ev t  -mod CXR --seed 30


#CutPaste: localization
python UPD_study/models/CutPaste/CPtrainer.py -lp t  -mod MRI --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t  -mod MRI --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -lp t  -mod MRI --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -lp t  -mod MRI -seq t1 --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t  -mod MRI -seq t1 --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -lp t  -mod MRI -seq t1 --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -lp t  -mod MRI --brats_t1 f -seq t1 --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t  -mod MRI --brats_t1 f -seq t1 --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -lp t  -mod MRI --brats_t1 f -seq t1 --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -lp t  -mod RF --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t  -mod RF --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -lp t  -mod RF --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -lp t  -mod CXR --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t  -mod CXR --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -lp t  -mod CXR --seed 30

python UPD_study/models/CutPaste/CPtrainer.py -lp t -ev t -mod MRI --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -ev t  -mod MRI --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -ev t  -mod MRI --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -lp t -ev t  -mod MRI -seq t1 --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -ev t  -mod MRI -seq t1 --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -ev t  -mod MRI -seq t1 --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -lp t -ev t  -mod MRI --brats_t1 f -seq t1 --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -ev t  -mod MRI --brats_t1 f -seq t1 --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -ev t  -mod MRI --brats_t1 f -seq t1 --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -lp t -ev t  -mod RF --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -ev t  -mod RF --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -ev t  -mod RF --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -lp t -ev t  -mod CXR --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -ev t  -mod CXR --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -ev t  -mod CXR --seed 30

#CutPaste: image level
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f  -mod MRI --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f  -mod MRI --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f  -mod MRI --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f  -mod MRI -seq t1 --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f  -mod MRI -seq t1 --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f  -mod MRI -seq t1 --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f  -mod MRI --brats_t1 f -seq t1 --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f  -mod MRI --brats_t1 f -seq t1 --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f  -mod MRI --brats_t1 f -seq t1 --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f  -mod RF --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f  -mod RF --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f  -mod RF --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f  -mod CXR --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f  -mod CXR --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f  -mod CXR --seed 30

python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f -ev t -mod MRI --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f -ev t  -mod MRI --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f -ev t  -mod MRI --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f -ev t  -mod MRI -seq t1 --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f -ev t  -mod MRI -seq t1 --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f -ev t  -mod MRI -seq t1 --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f -ev t  -mod MRI --brats_t1 f -seq t1 --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f -ev t  -mod MRI --brats_t1 f -seq t1 --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f -ev t  -mod MRI --brats_t1 f -seq t1 --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f -ev t  -mod RF --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f -ev t  -mod RF --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f -ev t  -mod RF --seed 30
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f -ev t  -mod CXR --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f -ev t  -mod CXR --seed 20 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f -ev t  -mod CXR --seed 30