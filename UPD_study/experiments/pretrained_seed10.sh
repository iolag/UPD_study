# # Activate the virtual environment
# source /opt/anaconda3/etc/profile.d/conda.sh
# conda activate upd



# Pre-training backbones with CCD 

python UPD_study/models/CCD/CCDtrainer.py -arch resnet18 -mod MRI 
python UPD_study/models/CCD/CCDtrainer.py -arch wide_resnet50_2 -mod MRI 
python UPD_study/models/CCD/CCDtrainer.py -arch vae -mod MRI 
python UPD_study/models/CCD/CCDtrainer.py -arch fanogan -mod MRI 
#python UPD_study/models/CCD/CCDtrainer.py -arch vgg19 -mod MRI 
python UPD_study/models/CCD/CCDtrainer.py -arch unet -mod MRI 
python UPD_study/models/CCD/CCDtrainer.py -arch pii -mod MRI 
python UPD_study/models/CCD/CCDtrainer.py -arch amc -mod MRI 

python UPD_study/models/CCD/CCDtrainer.py -arch resnet18 -mod MRI -seq t1 
python UPD_study/models/CCD/CCDtrainer.py -arch wide_resnet50_2 -mod MRI -seq t1 
python UPD_study/models/CCD/CCDtrainer.py -arch vae -mod MRI -seq t1 
python UPD_study/models/CCD/CCDtrainer.py -arch fanogan -mod MRI -seq t1 
#python UPD_study/models/CCD/CCDtrainer.py -arch vgg19 -mod MRI -seq t1 
python UPD_study/models/CCD/CCDtrainer.py -arch unet -mod MRI -seq t1 
python UPD_study/models/CCD/CCDtrainer.py -arch pii -mod MRI -seq t1 
python UPD_study/models/CCD/CCDtrainer.py -arch amc -mod MRI -seq t1

python UPD_study/models/CCD/CCDtrainer.py -arch resnet18 -mod CXR 
python UPD_study/models/CCD/CCDtrainer.py -arch wide_resnet50_2 -mod CXR 
python UPD_study/models/CCD/CCDtrainer.py -arch vae -mod CXR 
python UPD_study/models/CCD/CCDtrainer.py -arch fanogan -mod CXR 
#python UPD_study/models/CCD/CCDtrainer.py -arch vgg19 -mod CXR 
python UPD_study/models/CCD/CCDtrainer.py -arch unet -mod CXR 
python UPD_study/models/CCD/CCDtrainer.py -arch pii -mod CXR 
python UPD_study/models/CCD/CCDtrainer.py -arch amc -mod CXR

python UPD_study/models/CCD/CCDtrainer.py -arch resnet18 -mod RF 
python UPD_study/models/CCD/CCDtrainer.py -arch wide_resnet50_2 -mod RF 
python UPD_study/models/CCD/CCDtrainer.py -arch vae -mod RF 
python UPD_study/models/CCD/CCDtrainer.py -arch fanogan -mod RF 
#python UPD_study/models/CCD/CCDtrainer.py -arch vgg19 -mod RF 
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

# Method training
python UPD_study/models/AMCons/AMCtrainer.py -lp t -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py -lp t -mod MRI  
#python UPD_study/models/DFR/DFRtrainer.py -lp t -mod MRI  
python UPD_study/models/FAE/FAEtrainer.py -lp t -mod MRI  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t -mod MRI
python UPD_study/models/PADIM/PADIMtrainer.py -lp t -mod MRI  
python UPD_study/models/PII/PIItrainer.py -lp t -mod MRI  
python UPD_study/models/RD/RDtrainer.py -lp t -mod MRI  
python UPD_study/models/VAE/VAEtrainer.py -lp t -mod MRI  

python UPD_study/models/AMCons/AMCtrainer.py -lp t -mod MRI -seq t1 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t -mod MRI -seq t1  
python UPD_study/models/DAE/DAEtrainer.py -lp t -mod MRI -seq t1  
#python UPD_study/models/DFR/DFRtrainer.py -lp t -mod MRI -seq t1  
python UPD_study/models/FAE/FAEtrainer.py -lp t -mod MRI -seq t1  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t -mod MRI -seq t1
python UPD_study/models/PADIM/PADIMtrainer.py -lp t -mod MRI -seq t1  
python UPD_study/models/PII/PIItrainer.py -lp t -mod MRI -seq t1  
python UPD_study/models/RD/RDtrainer.py -lp t -mod MRI -seq t1  
python UPD_study/models/VAE/VAEtrainer.py -lp t -mod MRI -seq t1

python UPD_study/models/AMCons/AMCtrainer.py -lp t -mod CXR 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t -mod CXR  
python UPD_study/models/DAE/DAEtrainer.py -lp t -mod CXR  
#python UPD_study/models/DFR/DFRtrainer.py -lp t -mod CXR  
python UPD_study/models/FAE/FAEtrainer.py -lp t -mod CXR  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t -mod CXR
python UPD_study/models/PADIM/PADIMtrainer.py -lp t -mod CXR  
python UPD_study/models/PII/PIItrainer.py -lp t -mod CXR  
python UPD_study/models/RD/RDtrainer.py -lp t -mod CXR  
python UPD_study/models/VAE/VAEtrainer.py -lp t -mod CXR  

python UPD_study/models/AMCons/AMCtrainer.py -lp t -mod RF 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t -mod RF  
python UPD_study/models/DAE/DAEtrainer.py -lp t -mod RF  
#python UPD_study/models/DFR/DFRtrainer.py -lp t -mod RF  
python UPD_study/models/FAE/FAEtrainer.py -lp t -mod RF  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t -mod RF
python UPD_study/models/PADIM/PADIMtrainer.py -lp t -mod RF  
python UPD_study/models/PII/PIItrainer.py -lp t -mod RF  
python UPD_study/models/RD/RDtrainer.py -lp t -mod RF  
python UPD_study/models/VAE/VAEtrainer.py -lp t -mod RF  

# Method evaluation
python UPD_study/models/AMCons/AMCtrainer.py -lp t -ev t -mod MRI 
python UPD_study/models/CFLOW-AD/CFtrainer.py -lp t -ev t -mod MRI  
python UPD_study/models/DAE/DAEtrainer.py -lp t -ev t -mod MRI  
#python UPD_study/models/DFR/DFRtrainer.py -lp t -ev t -mod MRI  
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
#python UPD_study/models/DFR/DFRtrainer.py -lp t -ev t -mod MRI -seq t1  
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
#python UPD_study/models/DFR/DFRtrainer.py -lp t -ev t -mod MRI -seq t1 --brats_t1 f  
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
#python UPD_study/models/DFR/DFRtrainer.py -lp t -ev t -mod CXR  
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
#python UPD_study/models/DFR/DFRtrainer.py -lp t -ev t -mod RF  
python UPD_study/models/FAE/FAEtrainer.py -lp t -ev t -mod RF  
python UPD_study/models/f-AnoGAN/GANtrainer.py -lp t -ev t -mod RF  
python UPD_study/models/PADIM/PADIMtrainer.py -lp t -ev t -mod RF  
python UPD_study/models/PII/PIItrainer.py -lp t -ev t -mod RF  
python UPD_study/models/RD/RDtrainer.py -lp t -ev t -mod RF  
python UPD_study/models/VAE/VAEtrainer.py -lp t -ev t -mod RF  
python UPD_study/models/VAE/r-VAEtrainer.py -lp t -ev t -mod RF  



#newly added methods: expVAE, CutPaste

#expVAE

python UPD_study/models/expVAE/expVAEtrainer.py -lp t  -mod MRI --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t  -mod MRI -seq t1 --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t  -mod MRI --brats_t1 f -seq t1 --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t  -mod RF --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t  -mod CXR --seed 10 


python UPD_study/models/expVAE/expVAEtrainer.py -lp t -ev t -mod MRI --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t -ev t  -mod MRI -seq t1 --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t -ev t  -mod MRI --brats_t1 f -seq t1 --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t -ev t  -mod RF --seed 10 
python UPD_study/models/expVAE/expVAEtrainer.py -lp t -ev t  -mod CXR --seed 10 



#CutPaste: localization
python UPD_study/models/CutPaste/CPtrainer.py -lp t  -mod MRI --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t  -mod MRI -seq t1 --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t  -mod MRI --brats_t1 f -seq t1 --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t  -mod RF --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t  -mod CXR --seed 10 

python UPD_study/models/CutPaste/CPtrainer.py -lp t -ev t -mod MRI --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -ev t  -mod MRI -seq t1 --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -ev t  -mod MRI --brats_t1 f -seq t1 --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -ev t  -mod RF --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -ev t  -mod CXR --seed 10 

#CutPaste: image level
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f  -mod MRI --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f  -mod MRI -seq t1 --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f  -mod MRI --brats_t1 f -seq t1 --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f  -mod RF --seed 10 
python UPD_study/models/CutPaste/CPtrainer.py -lp t -loc f  -mod CXR --seed 10 