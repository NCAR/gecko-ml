#!/bin/bash -l
#PBS -N gecko-ml
#PBS -A NAML0001
#PBS -l walltime=00:30:00
#PBS -q casper
#PBS -l select=1:ncpus=16:mpiprocs=16:mem=128GB
#PBS -o ./run.txt
#PBS -e ./run.txt
module load cuda/11 cudnn
conda activate gecko
cd /glade/work/$USER/gecko-ml/applications/
python run_gecko_emulators.py -c ../config/apin_O3_agg.yml >& ./run.txt
