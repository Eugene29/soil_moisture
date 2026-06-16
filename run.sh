#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prithvi

cd /home/yjean234/Azad/soil_moisture

day=$(date +%F)
log="log/$day"
mkdir -p $log
time=$(date +%s)
echo Writing log in direcotry: $log...

python train.py --modality both  --t-hls 1 --t-merra 1 --seed 0 --dev 0 &> $log/${time}0.txt &
# python train.py --modality hls   --t-hls 1             --seed 0 --dev 1 &> $log/${time}1.txt # &
# python train.py --modality hls   --t-hls 1             --seed 0 --dev 2 &> $log/${time}1.txt # &
# python train.py --modality merra           --t-merra 1 --seed 0 --dev 2 &> $log/${time}2.txt &
wait