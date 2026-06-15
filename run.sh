#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prithvi

cd /home/yjean234/Azad/soil_moisture
mkdir -p log

python train.py --modality both  --t-hls 1 --t-merra 1 --seed 0 --dev 0 &> log/0.txt &
python train.py --modality hls   --t-hls 1             --seed 0 --dev 1 &> log/1.txt &
python train.py --modality merra           --t-merra 1 --seed 0 --dev 2 &> log/2.txt &
wait