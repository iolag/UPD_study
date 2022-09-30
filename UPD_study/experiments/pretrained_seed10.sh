# Activate the virtual environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lagi

# seed 10

# Pre-training backbones with CCD 
python models/CCD/CCDtrainer.py -arch resnet18 -mod MRI 
python models/CCD/CCDtrainer.py -arch wide_resnet50_2 -mod MRI 
python models/CCD/CCDtrainer.py -arch vae -mod MRI 
python models/CCD/CCDtrainer.py -arch fanogan -mod MRI 
python models/CCD/CCDtrainer.py -arch vgg19 -mod MRI 
python models/CCD/CCDtrainer.py -arch unet -mod MRI 
python models/CCD/CCDtrainer.py -arch pii -mod MRI 
python models/CCD/CCDtrainer.py -arch amc -mod MRI 

python models/CCD/CCDtrainer.py -arch resnet18 -mod MRI -seq t1 
python models/CCD/CCDtrainer.py -arch wide_resnet50_2 -mod MRI -seq t1 
python models/CCD/CCDtrainer.py -arch vae -mod MRI -seq t1 
python models/CCD/CCDtrainer.py -arch fanogan -mod MRI -seq t1 
python models/CCD/CCDtrainer.py -arch vgg19 -mod MRI -seq t1 
python models/CCD/CCDtrainer.py -arch unet -mod MRI -seq t1 
python models/CCD/CCDtrainer.py -arch pii -mod MRI -seq t1 
python models/CCD/CCDtrainer.py -arch amc -mod MRI -seq t1

python models/CCD/CCDtrainer.py -arch resnet18 -mod CXR 
python models/CCD/CCDtrainer.py -arch wide_resnet50_2 -mod CXR 
python models/CCD/CCDtrainer.py -arch vae -mod CXR 
python models/CCD/CCDtrainer.py -arch fanogan -mod CXR 
python models/CCD/CCDtrainer.py -arch vgg19 -mod CXR 
python models/CCD/CCDtrainer.py -arch unet -mod CXR 
python models/CCD/CCDtrainer.py -arch pii -mod CXR 
python models/CCD/CCDtrainer.py -arch amc -mod CXR

python models/CCD/CCDtrainer.py -arch resnet18 -mod RF 
python models/CCD/CCDtrainer.py -arch wide_resnet50_2 -mod RF 
python models/CCD/CCDtrainer.py -arch vae -mod RF 
python models/CCD/CCDtrainer.py -arch fanogan -mod RF 
python models/CCD/CCDtrainer.py -arch vgg19 -mod RF 
python models/CCD/CCDtrainer.py -arch unet -mod RF 
python models/CCD/CCDtrainer.py -arch pii -mod RF 
python models/CCD/CCDtrainer.py -arch amc -mod RF

# Method training
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

# Method evaluation
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

# seed 20

# Pre-training backbones with CCD 
python models/CCD/CCDtrainer.py --seed 20 -arch resnet18 -mod MRI 
python models/CCD/CCDtrainer.py --seed 20 -arch wide_resnet50_2 -mod MRI 
python models/CCD/CCDtrainer.py --seed 20 -arch vae -mod MRI 
python models/CCD/CCDtrainer.py --seed 20 -arch fanogan -mod MRI 
python models/CCD/CCDtrainer.py --seed 20 -arch vgg19 -mod MRI 
python models/CCD/CCDtrainer.py --seed 20 -arch unet -mod MRI 
python models/CCD/CCDtrainer.py --seed 20 -arch pii -mod MRI 
python models/CCD/CCDtrainer.py --seed 20 -arch amc -mod MRI 

python models/CCD/CCDtrainer.py --seed 20 -arch resnet18 -mod MRI -seq t1 
python models/CCD/CCDtrainer.py --seed 20 -arch wide_resnet50_2 -mod MRI -seq t1 
python models/CCD/CCDtrainer.py --seed 20 -arch vae -mod MRI -seq t1 
python models/CCD/CCDtrainer.py --seed 20 -arch fanogan -mod MRI -seq t1 
python models/CCD/CCDtrainer.py --seed 20 -arch vgg19 -mod MRI -seq t1 
python models/CCD/CCDtrainer.py --seed 20 -arch unet -mod MRI -seq t1 
python models/CCD/CCDtrainer.py --seed 20 -arch pii -mod MRI -seq t1 
python models/CCD/CCDtrainer.py --seed 20 -arch amc -mod MRI -seq t1

python models/CCD/CCDtrainer.py --seed 20 -arch resnet18 -mod CXR 
python models/CCD/CCDtrainer.py --seed 20 -arch wide_resnet50_2 -mod CXR 
python models/CCD/CCDtrainer.py --seed 20 -arch vae -mod CXR 
python models/CCD/CCDtrainer.py --seed 20 -arch fanogan -mod CXR 
python models/CCD/CCDtrainer.py --seed 20 -arch vgg19 -mod CXR 
python models/CCD/CCDtrainer.py --seed 20 -arch unet -mod CXR 
python models/CCD/CCDtrainer.py --seed 20 -arch pii -mod CXR 
python models/CCD/CCDtrainer.py --seed 20 -arch amc -mod CXR

python models/CCD/CCDtrainer.py --seed 20 -arch resnet18 -mod RF 
python models/CCD/CCDtrainer.py --seed 20 -arch wide_resnet50_2 -mod RF 
python models/CCD/CCDtrainer.py --seed 20 -arch vae -mod RF 
python models/CCD/CCDtrainer.py --seed 20 -arch fanogan -mod RF 
python models/CCD/CCDtrainer.py --seed 20 -arch vgg19 -mod RF 
python models/CCD/CCDtrainer.py --seed 20 -arch unet -mod RF 
python models/CCD/CCDtrainer.py --seed 20 -arch pii -mod RF 
python models/CCD/CCDtrainer.py --seed 20 -arch amc -mod RF

# Method training
python models/AMCons/AMCtrainer.py --seed 20 -lp t -mod MRI 
python models/CFLOW-AD/CFtrainer.py --seed 20 -lp t -mod MRI  
python models/DAE/DAEtrainer.py --seed 20 -lp t -mod MRI  
python models/DFR/DFRtrainer.py --seed 20 -lp t -mod MRI  
python models/FAE/FAEtrainer.py --seed 20 -lp t -mod MRI  
python models/f-AnoGAN/GANtrainer.py --seed 20 -lp t -mod MRI
python models/PADIM/PADIMtrainer.py --seed 20 -lp t -mod MRI  
python models/PII/PIItrainer.py --seed 20 -lp t -mod MRI  
python models/RD/RDtrainer.py --seed 20 -lp t -mod MRI  
python models/VAE/VAEtrainer.py --seed 20 -lp t -mod MRI  

python models/AMCons/AMCtrainer.py --seed 20 -lp t -mod MRI -seq t1 
python models/CFLOW-AD/CFtrainer.py --seed 20 -lp t -mod MRI -seq t1  
python models/DAE/DAEtrainer.py --seed 20 -lp t -mod MRI -seq t1  
python models/DFR/DFRtrainer.py --seed 20 -lp t -mod MRI -seq t1  
python models/FAE/FAEtrainer.py --seed 20 -lp t -mod MRI -seq t1  
python models/f-AnoGAN/GANtrainer.py --seed 20 -lp t -mod MRI -seq t1
python models/PADIM/PADIMtrainer.py --seed 20 -lp t -mod MRI -seq t1  
python models/PII/PIItrainer.py --seed 20 -lp t -mod MRI -seq t1  
python models/RD/RDtrainer.py --seed 20 -lp t -mod MRI -seq t1  
python models/VAE/VAEtrainer.py --seed 20 -lp t -mod MRI -seq t1

python models/AMCons/AMCtrainer.py --seed 20 -lp t -mod CXR 
python models/CFLOW-AD/CFtrainer.py --seed 20 -lp t -mod CXR  
python models/DAE/DAEtrainer.py --seed 20 -lp t -mod CXR  
python models/DFR/DFRtrainer.py --seed 20 -lp t -mod CXR  
python models/FAE/FAEtrainer.py --seed 20 -lp t -mod CXR  
python models/f-AnoGAN/GANtrainer.py --seed 20 -lp t -mod CXR
python models/PADIM/PADIMtrainer.py --seed 20 -lp t -mod CXR  
python models/PII/PIItrainer.py --seed 20 -lp t -mod CXR  
python models/RD/RDtrainer.py --seed 20 -lp t -mod CXR  
python models/VAE/VAEtrainer.py --seed 20 -lp t -mod CXR  

python models/AMCons/AMCtrainer.py --seed 20 -lp t -mod RF 
python models/CFLOW-AD/CFtrainer.py --seed 20 -lp t -mod RF  
python models/DAE/DAEtrainer.py --seed 20 -lp t -mod RF  
python models/DFR/DFRtrainer.py --seed 20 -lp t -mod RF  
python models/FAE/FAEtrainer.py --seed 20 -lp t -mod RF  
python models/f-AnoGAN/GANtrainer.py --seed 20 -lp t -mod RF
python models/PADIM/PADIMtrainer.py --seed 20 -lp t -mod RF  
python models/PII/PIItrainer.py --seed 20 -lp t -mod RF  
python models/RD/RDtrainer.py --seed 20 -lp t -mod RF  
python models/VAE/VAEtrainer.py --seed 20 -lp t -mod RF  

# Method evaluation
python models/AMCons/AMCtrainer.py --seed 20 -lp t -ev t -mod MRI 
python models/CFLOW-AD/CFtrainer.py --seed 20 -lp t -ev t -mod MRI  
python models/DAE/DAEtrainer.py --seed 20 -lp t -ev t -mod MRI  
python models/DFR/DFRtrainer.py --seed 20 -lp t -ev t -mod MRI  
python models/FAE/FAEtrainer.py --seed 20 -lp t -ev t -mod MRI  
python models/f-AnoGAN/GANtrainer.py --seed 20 -lp t -ev t -mod MRI  
python models/PADIM/PADIMtrainer.py --seed 20 -lp t -ev t -mod MRI  
python models/PII/PIItrainer.py --seed 20 -lp t -ev t -mod MRI  
python models/RD/RDtrainer.py --seed 20 -lp t -ev t -mod MRI  
python models/VAE/VAEtrainer.py --seed 20 -lp t -ev t -mod MRI  
python models/VAE/r-VAEtrainer.py --seed 20 -lp t -ev t -mod MRI 

python models/AMCons/AMCtrainer.py --seed 20 -lp t -ev t -mod MRI -seq t1 
python models/CFLOW-AD/CFtrainer.py --seed 20 -lp t -ev t -mod MRI -seq t1  
python models/DAE/DAEtrainer.py --seed 20 -lp t -ev t -mod MRI -seq t1  
python models/DFR/DFRtrainer.py --seed 20 -lp t -ev t -mod MRI -seq t1  
python models/FAE/FAEtrainer.py --seed 20 -lp t -ev t -mod MRI -seq t1  
python models/f-AnoGAN/GANtrainer.py --seed 20 -lp t -ev t -mod MRI -seq t1  
python models/PADIM/PADIMtrainer.py --seed 20 -lp t -ev t -mod MRI -seq t1  
python models/PII/PIItrainer.py --seed 20 -lp t -ev t -mod MRI -seq t1  
python models/RD/RDtrainer.py --seed 20 -lp t -ev t -mod MRI -seq t1  
python models/VAE/VAEtrainer.py --seed 20 -lp t -ev t -mod MRI -seq t1
python models/VAE/r-VAEtrainer.py --seed 20 -lp t -ev t -mod MRI -seq t1

python models/AMCons/AMCtrainer.py --seed 20 -lp t -ev t -mod MRI -seq t1 --brats_t1 f 
python models/CFLOW-AD/CFtrainer.py --seed 20 -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python models/DAE/DAEtrainer.py --seed 20 -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python models/DFR/DFRtrainer.py --seed 20 -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python models/FAE/FAEtrainer.py --seed 20 -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python models/f-AnoGAN/GANtrainer.py --seed 20 -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python models/PADIM/PADIMtrainer.py --seed 20 -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python models/PII/PIItrainer.py --seed 20 -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python models/RD/RDtrainer.py --seed 20 -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python models/VAE/VAEtrainer.py --seed 20 -lp t -ev t -mod MRI -seq t1 --brats_t1 f
python models/VAE/r-VAEtrainer.py --seed 20 -lp t -ev t -mod MRI -seq t1 --brats_t1 f

python models/AMCons/AMCtrainer.py --seed 20 -lp t -ev t -mod CXR 
python models/CFLOW-AD/CFtrainer.py --seed 20 -lp t -ev t -mod CXR  
python models/DAE/DAEtrainer.py --seed 20 -lp t -ev t -mod CXR  
python models/DFR/DFRtrainer.py --seed 20 -lp t -ev t -mod CXR  
python models/FAE/FAEtrainer.py --seed 20 -lp t -ev t -mod CXR  
python models/f-AnoGAN/GANtrainer.py --seed 20 -lp t -ev t -mod CXR  
python models/PADIM/PADIMtrainer.py --seed 20 -lp t -ev t -mod CXR  
python models/PII/PIItrainer.py --seed 20 -lp t -ev t -mod CXR  
python models/RD/RDtrainer.py --seed 20 -lp t -ev t -mod CXR  
python models/VAE/VAEtrainer.py --seed 20 -lp t -ev t -mod CXR  
python models/VAE/r-VAEtrainer.py --seed 20 -lp t -ev t -mod CXR

python models/AMCons/AMCtrainer.py --seed 20 -lp t -ev t -mod RF 
python models/CFLOW-AD/CFtrainer.py --seed 20 -lp t -ev t -mod RF  
python models/DAE/DAEtrainer.py --seed 20 -lp t -ev t -mod RF  
python models/DFR/DFRtrainer.py --seed 20 -lp t -ev t -mod RF  
python models/FAE/FAEtrainer.py --seed 20 -lp t -ev t -mod RF  
python models/f-AnoGAN/GANtrainer.py --seed 20 -lp t -ev t -mod RF  
python models/PADIM/PADIMtrainer.py --seed 20 -lp t -ev t -mod RF  
python models/PII/PIItrainer.py --seed 20 -lp t -ev t -mod RF  
python models/RD/RDtrainer.py --seed 20 -lp t -ev t -mod RF  
python models/VAE/VAEtrainer.py --seed 20 -lp t -ev t -mod RF  
python models/VAE/r-VAEtrainer.py --seed 20 -lp t -ev t -mod RF  

# seed 30

# Pre-training backbones with CCD 
python models/CCD/CCDtrainer.py --seed 30 -arch resnet18 -mod MRI 
python models/CCD/CCDtrainer.py --seed 30 -arch wide_resnet50_2 -mod MRI 
python models/CCD/CCDtrainer.py --seed 30 -arch vae -mod MRI 
python models/CCD/CCDtrainer.py --seed 30 -arch fanogan -mod MRI 
python models/CCD/CCDtrainer.py --seed 30 -arch vgg19 -mod MRI 
python models/CCD/CCDtrainer.py --seed 30 -arch unet -mod MRI 
python models/CCD/CCDtrainer.py --seed 30 -arch pii -mod MRI 
python models/CCD/CCDtrainer.py --seed 30 -arch amc -mod MRI 

python models/CCD/CCDtrainer.py --seed 30 -arch resnet18 -mod MRI -seq t1 
python models/CCD/CCDtrainer.py --seed 30 -arch wide_resnet50_2 -mod MRI -seq t1 
python models/CCD/CCDtrainer.py --seed 30 -arch vae -mod MRI -seq t1 
python models/CCD/CCDtrainer.py --seed 30 -arch fanogan -mod MRI -seq t1 
python models/CCD/CCDtrainer.py --seed 30 -arch vgg19 -mod MRI -seq t1 
python models/CCD/CCDtrainer.py --seed 30 -arch unet -mod MRI -seq t1 
python models/CCD/CCDtrainer.py --seed 30 -arch pii -mod MRI -seq t1 
python models/CCD/CCDtrainer.py --seed 30 -arch amc -mod MRI -seq t1

python models/CCD/CCDtrainer.py --seed 30 -arch resnet18 -mod CXR 
python models/CCD/CCDtrainer.py --seed 30 -arch wide_resnet50_2 -mod CXR 
python models/CCD/CCDtrainer.py --seed 30 -arch vae -mod CXR 
python models/CCD/CCDtrainer.py --seed 30 -arch fanogan -mod CXR 
python models/CCD/CCDtrainer.py --seed 30 -arch vgg19 -mod CXR 
python models/CCD/CCDtrainer.py --seed 30 -arch unet -mod CXR 
python models/CCD/CCDtrainer.py --seed 30 -arch pii -mod CXR 
python models/CCD/CCDtrainer.py --seed 30 -arch amc -mod CXR

python models/CCD/CCDtrainer.py --seed 30 -arch resnet18 -mod RF 
python models/CCD/CCDtrainer.py --seed 30 -arch wide_resnet50_2 -mod RF 
python models/CCD/CCDtrainer.py --seed 30 -arch vae -mod RF 
python models/CCD/CCDtrainer.py --seed 30 -arch fanogan -mod RF 
python models/CCD/CCDtrainer.py --seed 30 -arch vgg19 -mod RF 
python models/CCD/CCDtrainer.py --seed 30 -arch unet -mod RF 
python models/CCD/CCDtrainer.py --seed 30 -arch pii -mod RF 
python models/CCD/CCDtrainer.py --seed 30 -arch amc -mod RF

# Method training
python models/AMCons/AMCtrainer.py --seed 30 -lp t -mod MRI 
python models/CFLOW-AD/CFtrainer.py --seed 30 -lp t -mod MRI  
python models/DAE/DAEtrainer.py --seed 30 -lp t -mod MRI  
python models/DFR/DFRtrainer.py --seed 30 -lp t -mod MRI  
python models/FAE/FAEtrainer.py --seed 30 -lp t -mod MRI  
python models/f-AnoGAN/GANtrainer.py --seed 30 -lp t -mod MRI
python models/PADIM/PADIMtrainer.py --seed 30 -lp t -mod MRI  
python models/PII/PIItrainer.py --seed 30 -lp t -mod MRI  
python models/RD/RDtrainer.py --seed 30 -lp t -mod MRI  
python models/VAE/VAEtrainer.py --seed 30 -lp t -mod MRI  

python models/AMCons/AMCtrainer.py --seed 30 -lp t -mod MRI -seq t1 
python models/CFLOW-AD/CFtrainer.py --seed 30 -lp t -mod MRI -seq t1  
python models/DAE/DAEtrainer.py --seed 30 -lp t -mod MRI -seq t1  
python models/DFR/DFRtrainer.py --seed 30 -lp t -mod MRI -seq t1  
python models/FAE/FAEtrainer.py --seed 30 -lp t -mod MRI -seq t1  
python models/f-AnoGAN/GANtrainer.py --seed 30 -lp t -mod MRI -seq t1
python models/PADIM/PADIMtrainer.py --seed 30 -lp t -mod MRI -seq t1  
python models/PII/PIItrainer.py --seed 30 -lp t -mod MRI -seq t1  
python models/RD/RDtrainer.py --seed 30 -lp t -mod MRI -seq t1  
python models/VAE/VAEtrainer.py --seed 30 -lp t -mod MRI -seq t1

python models/AMCons/AMCtrainer.py --seed 30 -lp t -mod CXR 
python models/CFLOW-AD/CFtrainer.py --seed 30 -lp t -mod CXR  
python models/DAE/DAEtrainer.py --seed 30 -lp t -mod CXR  
python models/DFR/DFRtrainer.py --seed 30 -lp t -mod CXR  
python models/FAE/FAEtrainer.py --seed 30 -lp t -mod CXR  
python models/f-AnoGAN/GANtrainer.py --seed 30 -lp t -mod CXR
python models/PADIM/PADIMtrainer.py --seed 30 -lp t -mod CXR  
python models/PII/PIItrainer.py --seed 30 -lp t -mod CXR  
python models/RD/RDtrainer.py --seed 30 -lp t -mod CXR  
python models/VAE/VAEtrainer.py --seed 30 -lp t -mod CXR  

python models/AMCons/AMCtrainer.py --seed 30 -lp t -mod RF 
python models/CFLOW-AD/CFtrainer.py --seed 30 -lp t -mod RF  
python models/DAE/DAEtrainer.py --seed 30 -lp t -mod RF  
python models/DFR/DFRtrainer.py --seed 30 -lp t -mod RF  
python models/FAE/FAEtrainer.py --seed 30 -lp t -mod RF  
python models/f-AnoGAN/GANtrainer.py --seed 30 -lp t -mod RF
python models/PADIM/PADIMtrainer.py --seed 30 -lp t -mod RF  
python models/PII/PIItrainer.py --seed 30 -lp t -mod RF  
python models/RD/RDtrainer.py --seed 30 -lp t -mod RF  
python models/VAE/VAEtrainer.py --seed 30 -lp t -mod RF  

# Method evaluation
python models/AMCons/AMCtrainer.py --seed 30 -lp t -ev t -mod MRI 
python models/CFLOW-AD/CFtrainer.py --seed 30 -lp t -ev t -mod MRI  
python models/DAE/DAEtrainer.py --seed 30 -lp t -ev t -mod MRI  
python models/DFR/DFRtrainer.py --seed 30 -lp t -ev t -mod MRI  
python models/FAE/FAEtrainer.py --seed 30 -lp t -ev t -mod MRI  
python models/f-AnoGAN/GANtrainer.py --seed 30 -lp t -ev t -mod MRI  
python models/PADIM/PADIMtrainer.py --seed 30 -lp t -ev t -mod MRI  
python models/PII/PIItrainer.py --seed 30 -lp t -ev t -mod MRI  
python models/RD/RDtrainer.py --seed 30 -lp t -ev t -mod MRI  
python models/VAE/VAEtrainer.py --seed 30 -lp t -ev t -mod MRI  
python models/VAE/r-VAEtrainer.py --seed 30 -lp t -ev t -mod MRI 

python models/AMCons/AMCtrainer.py --seed 30 -lp t -ev t -mod MRI -seq t1 
python models/CFLOW-AD/CFtrainer.py --seed 30 -lp t -ev t -mod MRI -seq t1  
python models/DAE/DAEtrainer.py --seed 30 -lp t -ev t -mod MRI -seq t1  
python models/DFR/DFRtrainer.py --seed 30 -lp t -ev t -mod MRI -seq t1  
python models/FAE/FAEtrainer.py --seed 30 -lp t -ev t -mod MRI -seq t1  
python models/f-AnoGAN/GANtrainer.py --seed 30 -lp t -ev t -mod MRI -seq t1  
python models/PADIM/PADIMtrainer.py --seed 30 -lp t -ev t -mod MRI -seq t1  
python models/PII/PIItrainer.py --seed 30 -lp t -ev t -mod MRI -seq t1  
python models/RD/RDtrainer.py --seed 30 -lp t -ev t -mod MRI -seq t1  
python models/VAE/VAEtrainer.py --seed 30 -lp t -ev t -mod MRI -seq t1
python models/VAE/r-VAEtrainer.py --seed 30 -lp t -ev t -mod MRI -seq t1

python models/AMCons/AMCtrainer.py --seed 30 -lp t -ev t -mod MRI -seq t1 --brats_t1 f 
python models/CFLOW-AD/CFtrainer.py --seed 30 -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python models/DAE/DAEtrainer.py --seed 30 -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python models/DFR/DFRtrainer.py --seed 30 -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python models/FAE/FAEtrainer.py --seed 30 -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python models/f-AnoGAN/GANtrainer.py --seed 30 -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python models/PADIM/PADIMtrainer.py --seed 30 -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python models/PII/PIItrainer.py --seed 30 -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python models/RD/RDtrainer.py --seed 30 -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
python models/VAE/VAEtrainer.py --seed 30 -lp t -ev t -mod MRI -seq t1 --brats_t1 f
python models/VAE/r-VAEtrainer.py --seed 30 -lp t -ev t -mod MRI -seq t1 --brats_t1 f

python models/AMCons/AMCtrainer.py --seed 30 -lp t -ev t -mod CXR 
python models/CFLOW-AD/CFtrainer.py --seed 30 -lp t -ev t -mod CXR  
python models/DAE/DAEtrainer.py --seed 30 -lp t -ev t -mod CXR  
python models/DFR/DFRtrainer.py --seed 30 -lp t -ev t -mod CXR  
python models/FAE/FAEtrainer.py --seed 30 -lp t -ev t -mod CXR  
python models/f-AnoGAN/GANtrainer.py --seed 30 -lp t -ev t -mod CXR  
python models/PADIM/PADIMtrainer.py --seed 30 -lp t -ev t -mod CXR  
python models/PII/PIItrainer.py --seed 30 -lp t -ev t -mod CXR  
python models/RD/RDtrainer.py --seed 30 -lp t -ev t -mod CXR  
python models/VAE/VAEtrainer.py --seed 30 -lp t -ev t -mod CXR  
python models/VAE/r-VAEtrainer.py --seed 30 -lp t -ev t -mod CXR

python models/AMCons/AMCtrainer.py --seed 30 -lp t -ev t -mod RF 
python models/CFLOW-AD/CFtrainer.py --seed 30 -lp t -ev t -mod RF  
python models/DAE/DAEtrainer.py --seed 30 -lp t -ev t -mod RF  
python models/DFR/DFRtrainer.py --seed 30 -lp t -ev t -mod RF  
python models/FAE/FAEtrainer.py --seed 30 -lp t -ev t -mod RF  
python models/f-AnoGAN/GANtrainer.py --seed 30 -lp t -ev t -mod RF  
python models/PADIM/PADIMtrainer.py --seed 30 -lp t -ev t -mod RF  
python models/PII/PIItrainer.py --seed 30 -lp t -ev t -mod RF  
python models/RD/RDtrainer.py --seed 30 -lp t -ev t -mod RF  
python models/VAE/VAEtrainer.py --seed 30 -lp t -ev t -mod RF  
python models/VAE/r-VAEtrainer.py --seed 30 -lp t -ev t -mod RF  
