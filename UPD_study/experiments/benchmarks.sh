# Activate the virtual environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lagi

python UPD_study/models/AMCons/AMCtrainer.py -speed t
python UPD_study/models/AMCons/AMCtrainer.py -space t
python UPD_study/models/CFLOW-AD/CFtrainer.py -speed t
python UPD_study/models/CFLOW-AD/CFtrainer.py -space t
python UPD_study/models/DAE/DAEtrainer.py -speed t
python UPD_study/models/DAE/DAEtrainer.py -space t
python UPD_study/models/DFR/DFRtrainer.py -speed t
python UPD_study/models/DFR/DFRtrainer.py -space t
python UPD_study/models/FAE/FAEtrainer.py -speed t
python UPD_study/models/FAE/FAEtrainer.py -space t
python UPD_study/models/f-AnoGAN/GANtrainer.py -speed t
python UPD_study/models/f-AnoGAN/GANtrainer.py -space t
python UPD_study/models/PADIM/PADIMtrainer.py -speed t
python UPD_study/models/PADIM/PADIMtrainer.py -space t
python UPD_study/models/PII/PIItrainer.py -speed t
python UPD_study/models/PII/PIItrainer.py -space t
python UPD_study/models/RD/RDtrainer.py -speed t
python UPD_study/models/RD/RDtrainer.py -space t
python UPD_study/models/VAE/VAEtrainer.py -speed t
python UPD_study/models/VAE/VAEtrainer.py -space t
python UPD_study/models/VAE/r-VAEtrainer.py -speed t
python UPD_study/models/VAE/r-VAEtrainer.py -space t
