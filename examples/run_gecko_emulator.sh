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
module load ncarenv/1.3 gnu/8.3.0 openmpi/3.1.4 cuda/10.0 python/3.7.5
ncar_pylib ncar_20191211
cd /glade/work/$USER/gecko-ml/
python -u run_gecko_emulators.py -c ./config/dodecane_agg.yml >& ./run.txt
