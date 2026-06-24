#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prithvi

cd /home/yjean234/Azad/soil_moisture

python soil_moisture/scripts/train.py data=tx_2020 modality=both seed=0 dev=0
# python soil_moisture/scripts/train.py data=tx_2020 modality=both  T_HLS=1 T_MERRA=1 seed=0 dev=0 &
# python soil_moisture/scripts/train.py data=tx_2020 modality=hls   T_HLS=1           seed=0 dev=1 &
# python soil_moisture/scripts/train.py data=tx_2020 modality=merra         T_MERRA=1 seed=0 dev=2 &

wait