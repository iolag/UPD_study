# Activate the virtual environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lagi

python models/AMCons/AMCtrainer.py -speed t
python models/AMCons/AMCtrainer.py -space t
python models/CFLOW-AD/CFtrainer.py -speed t
python models/CFLOW-AD/CFtrainer.py -space t
python models/DAE/DAEtrainer.py -speed t
python models/DAE/DAEtrainer.py -space t
python models/DFR/DFRtrainer.py -speed t
python models/DFR/DFRtrainer.py -space t
python models/FAE/FAEtrainer.py -speed t
python models/FAE/FAEtrainer.py -space t
python models/f-AnoGAN/GANtrainer.py -speed t
python models/f-AnoGAN/GANtrainer.py -space t
python models/PADIM/PADIMtrainer.py -speed t
python models/PADIM/PADIMtrainer.py -space t
python models/PII/PIItrainer.py -speed t
python models/PII/PIItrainer.py -space t
python models/RD/RDtrainer.py -speed t
python models/RD/RDtrainer.py -space t
python models/VAE/VAEtrainer.py -speed t
python models/VAE/VAEtrainer.py -space t
python models/VAE/r-VAEtrainer.py -speed t
python models/VAE/r-VAEtrainer.py -space t