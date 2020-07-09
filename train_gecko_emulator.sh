#!/bin/bash -l
#SBATCH --job-name=gecko-ml
#SBATCH --account=NAML0001
#SBATCH --ntasks=36
#SBATCH --time=00:20:00
#SBATCH --partition=dav
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=128G
#SBATCH -o gecko_ml.out
#SBATCH -e gecko_ml.out
module load ncarenv/1.3 gnu/8.3.0 openmpi/3.1.4 python/3.7.5 cuda/10.1
ncar_pylib ncar_20191211
cd /glade/work/$USER/gecko-ml/
python -u train_gecko_emulators.py -c ./config/agg_config.yml >& gecko_train.txt
