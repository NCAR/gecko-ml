#!/bin/bash -l
#PBS -N gecko-ml
#PBS -A NAML0001
#PBS -l walltime=01:30:00
#PBS -q casper
#PBS -l select=1:ncpus=16:ngpus=1:mem=128GB -l gpu_type=v100
#PBS -o ./train.txt
#PBS -e ./train.txt
module load cuda/11 cudnn
conda activate gecko
cd /glade/work/$USER/gecko-ml/applications/
python train_gecko_emulators.py -c ../config/toluene_agg.yml >& ./train.txt
