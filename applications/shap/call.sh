#!/bin/bash

"""
    MLP models 
"""


python gecko_shap.py /glade/work/keelyl/geckonew/gecko-ml/config/toluene_agg.yml -s toluene/ -m /glade/work/keelyl/geckonew/gecko-ml/toluene_agg_runs_unvaried/4_27_models/toluene_dnn_1_20/

python gecko_shap.py /glade/work/keelyl/geckonew/gecko-ml/config/dodecane_agg.yml -s dodecane/ -m /glade/work/keelyl/geckonew/gecko-ml/dodecane_agg_runs/05_06_models/dodecane_dnn_1_3/

python gecko_shap.py /glade/work/keelyl/geckonew/gecko-ml/config/apin_O3_agg.yml -s alpha_pinene/ -m /glade/work/keelyl/geckonew/gecko-ml/apin_agg_runs/05_14_models/apin_O3_dnn_1_0/


"""
    GRU models
"""

python gecko_shap.py /glade/work/schreck/repos/GECKO_OPT/gecko-ml/echo/toluene/6/model.yml -s toluene_gru -m /glade/work/schreck/repos/GECKO_OPT/gecko-ml/echo/toluene/6/best.pt

python gecko_shap.py /glade/work/schreck/repos/GECKO_OPT/gecko-ml/echo/apin/0/model.yml -s apin_gru -m /glade/work/schreck/repos/GECKO_OPT/gecko-ml/echo/apin/0/best.pt

python gecko_shap.py /glade/work/schreck/repos/GECKO_OPT/gecko-ml/echo/dodecane/0/model.yml -s dodecane_gru -m /glade/work/schreck/repos/GECKO_OPT/gecko-ml/echo/dodecane/0/best.pt