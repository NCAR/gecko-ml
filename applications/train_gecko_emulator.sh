#!/bin/bash -l
#PBS -N gecko-ml
#PBS -A NAML0001
#PBS -l walltime=00:30:00
#PBS -q casper
#PBS -l select=1:ncpus=8:ngpus=1:mem=64GB -l gpu_type=v100
#PBS -o ./train.txt
#PBS -e ./train.txt
module load cuda/11 cudnn
conda activate gecko
cd /glade/work/$USER/gecko-ml/applications/
python train_gecko_emulators.py -c ../config/toluene_agg.yml >& ./train.txt
