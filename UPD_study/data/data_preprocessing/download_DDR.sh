# Activate the virtual environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate lagi


# gdown https://drive.google.com/uc?id=1_gmRXJ3Wc9lNScbAlC3Jt1zOwpFHfDgE
# gdown https://drive.google.com/uc?id=14a5HcSTtGwEchKMpUGieuv0rNfm0Mbs9
# gdown https://drive.google.com/uc?id=10i8OzBuHqPICgrkxoGLpHPy_3nDg0buX
# gdown https://drive.google.com/uc?id=1-26gAtV2TsoLioT3BsL8S5sieHLW3z-c
# gdown https://drive.google.com/uc?id=1joj-KN556wtnzipQ54OpPBMXM-pW-oJ9
# gdown https://drive.google.com/uc?id=1rNV-yqdcG5j9ZA05G0lmiZUc3xK7YRFv
# gdown https://drive.google.com/uc?id=12U7sn-8UwBgeL5pY7EOYUJAyqjFxTHS6
# gdown https://drive.google.com/uc?id=1fJwdYVO8_lDeJI8ic1yr7sl5RtqYfxOW
# gdown https://drive.google.com/uc?id=1rrWddjsrmrwKQVCWU9x7onVkgMdtv0Wf
# gdown https://drive.google.com/uc?id=1RxRx_ImIDAmUp1h8BNaqVJ9LJ0jCmyua

gdown https://drive.google.com/drive/folders/1z6tSFmxW_aNayUqVxx6h6bY4kwGzUTEC --folder
cat DDR-dataset.zip* > DDR-dataset.zip
unzip DDR-dataset.zip -d UPD_study/data/datasets/RF/
rm DDR-dataset.zip
rm -r DDR-dataset.zip*
mv 'UPD_study/data/datasets/RF/DDR-dataset/lesion_segmentation/valid/segmentation label' UPD_study/data/datasets/RF/DDR-dataset/lesion_segmentation/valid/label
