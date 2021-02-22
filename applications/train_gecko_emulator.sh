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
module load cuda/11 cudnn
conda activate gecko
echo $PATH
which python
cd /glade/work/$USER/gecko-ml/applications/
python train_gecko_emulators.py -c ../config/dodecane_agg.yml >& ./train.txt
