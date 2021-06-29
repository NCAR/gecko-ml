#!/bin/bash -l 

#PBS -N jupyter_instance
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:ngpus=1:mem=128GB
#PBS -A NAML0001
#PBS -q casper

### Merge output and error files
#PBS -j oe
#PBS -k eod

source ~/.bashrc
export PATH="/glade/work/${USER}/py37/bin:$PATH"


conf=/glade/work/schreck/repos/GECKO_OPT/gecko-ml/echo/apin/0/model.yml
save=toluene_gru/
model=/glade/work/schreck/repos/GECKO_OPT/gecko-ml/echo/apin/0/best.pt
workers=1
worker=0

python gecko_shap.py /glade/work/schreck/repos/GECKO_OPT/gecko-ml/echo/apin/0/model.yml -s toluene_gru -m /glade/work/schreck/repos/GECKO_OPT/gecko-ml/echo/apin/0/best.pt

python gecko_shap.py $conf -s $save -m $model --workers $workers --worker $worker
