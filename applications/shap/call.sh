#!/bin/bash

"""
    MLP models 
"""


python gecko_shap.py /glade/work/keelyl/geckonew/gecko-ml/config/toluene_agg.yml -s /glade/work/schreck/repos/GECKO_OPT/clean/gecko-ml/applications/shap/toluene/mlp -m /glade/work/keelyl/geckonew/gecko-ml/toluene_agg_runs_unvaried/4_27_models/toluene_dnn_1_20/

python gecko_shap.py /glade/work/keelyl/geckonew/gecko-ml/config/dodecane_agg.yml -s /glade/work/schreck/repos/GECKO_OPT/clean/gecko-ml/applications/shap/dodecane/mlp -m /glade/work/keelyl/geckonew/gecko-ml/dodecane_agg_runs/05_06_models/dodecane_dnn_1_3/

python gecko_shap.py /glade/work/keelyl/geckonew/gecko-ml/config/apin_O3_agg.yml -s /glade/work/schreck/repos/GECKO_OPT/clean/gecko-ml/applications/shap/apin/mlp -m /glade/work/keelyl/geckonew/gecko-ml/apin_agg_runs/05_14_models/apin_O3_dnn_1_0/


"""
    GRU models
"""

python gecko_shap.py ../../config/toluene_agg.yml -s /glade/work/schreck/repos/GECKO_OPT/clean/gecko-ml/applications/shap/toluene/gru -m /glade/work/schreck/repos/GECKO_OPT/gecko-ml/echo/toluene/6/best.pt --workers 10

python gecko_shap.py ../../config/apin_O3_agg.yml -s /glade/work/schreck/repos/GECKO_OPT/clean/gecko-ml/applications/shap/apin/gru -m /glade/work/schreck/repos/GECKO_OPT/gecko-ml/echo/apin/7/best.pt --workers 10

python gecko_shap.py ../../config/dodecane_agg.yml -s /glade/work/schreck/repos/GECKO_OPT/clean/gecko-ml/applications/shap/dodecane/gru -m /glade/work/schreck/repos/GECKO_OPT/gecko-ml/echo/dodecane/3/best.pt --workers 10