#!/bin/bash -l 

#PBS -N jupyter_instance
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=8:ngpus=1:mem=128GB
#PBS -A NAML0001
#PBS -q casper

### Merge output and error files
#PBS -j oe
#PBS -k eod

source ~/.bashrc
export PATH="/glade/work/${USER}/py37/bin:$PATH"


conf=/glade/work/keelyl/geckonew/gecko-ml/config/toluene_agg.yml
save=toluene/
model=/glade/work/keelyl/geckonew/gecko-ml/toluene_agg_runs_unvaried/4_27_models/toluene_dnn_1_20/
workers=10
worker=0

python gecko_shap.py $conf -s $save -m $model --workers $workers --worker $worker