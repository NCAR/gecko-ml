#!/bin/bash -l
#SBATCH --job-name=gecko-ml
#SBATCH --account=NAML0001
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=dav
#SBATCH --mem=128G
#SBATCH -o gecko_ml.out
#SBATCH -e gecko_ml.out

module load ncarenv/1.3 gnu/8.3.0 openmpi/3.1.4 python/3.7.5 cuda/10.1
ncar_pylib ncar_20191211
cd ~/gecko-ml
python -u run_gecko_emulators.py -c ./config/config_gas_aerosol.yml >& run_gecko.txt
