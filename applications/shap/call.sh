#!/bin/bash

python gecko_shap.py /glade/work/keelyl/geckonew/gecko-ml/config/toluene_agg.yml -s toluene/ -m /glade/work/keelyl/geckonew/gecko-ml/toluene_agg_runs_unvaried/4_27_models/toluene_dnn_1_20/

python gecko_shap.py /glade/work/keelyl/geckonew/gecko-ml/config/dodecane_agg.yml -s dodecane/ -m /glade/work/keelyl/geckonew/gecko-ml/dodecane_agg_runs/05_06_models/dodecane_dnn_1_3/

python gecko_shap.py /glade/work/keelyl/geckonew/gecko-ml/config/apin_O3_agg.yml -s alpha_pinene/ -m /glade/work/keelyl/geckonew/gecko-ml/apin_agg_runs/05_14_models/apin_O3_dnn_1_0/