#!/bin/bash -l
#SBATCH --job-name=gecko-ml
#SBATCH --account=NAML0001
#SBATCH --ntasks=36
#SBATCH --time=00:30:00
#SBATCH --partition=dav
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=128G
#SBATCH -o ./train.txt
#SBATCH -e ./train.txt
module load ncarenv/1.3 gnu/8.3.0 openmpi/3.1.4 python/3.7.5 cuda/10.1
cd /glade/work/$USER/gecko-ml/applications/
ncar_pylib ncar_20191211
python train_gecko_emulators.py -c ../config/dodecane_agg.yml >& ./train.txt
