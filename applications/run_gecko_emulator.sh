#!/bin/bash -l
#SBATCH --job-name=gecko-ml
#SBATCH --account=NAML0001
#SBATCH --time=00:50:00
#SBATCH --partition=dav
#SBATCH --nodes=2
#SBATCH --ntasks=100
#SBATCH --ntasks-per-node=50
#SBATCH --mem=256G
#SBATCH -o ./run.txt
#SBATCH -e ./run.txt
module load cuda/11 cudnn
conda activate gecko
cd /glade/work/$USER/gecko-ml/applications/
python run_gecko_emulators.py -c ../config/dodecane_agg.yml >& ./run.txt
