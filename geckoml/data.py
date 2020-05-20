import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from glob import glob
from os.path import join, exists
from matplotlib.colors import LogNorm
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import GroupKFold
from tensorflow.keras.optimizers import Adam

#Summary File

summary_file = "/glade/scratch/siyuan/POSTPROCESSING/ML2019_toluene_kOH/Batch1_toluene/GAR0084_Exp_List_TEXT_PreCompile_BatchSubmit_v0.csv"
summary_data = pd.read_csv(summary_file, skiprows=3)
summary_data.columns = summary_data.columns.str.strip()
summary_data.columns
summary_data["idnum"] = summary_data.index

#Load Data 

def load_data(path, summary_file, species="toluene_kOH"):
    """
    Load a set of experiment files based on a summary file.
    
    Args:
        path: Path to the directory containing summary and experiment files
        summary_file: Name of the summary file (should not contain the path)
        species: Name of the precursor chemical species
    
    Returns:
        exp_data_merged: All time series data for every experiment in pandas DataFrame
        summary_data: Summary experiment data file in pandas DataFrame
    """
    summary_data = pd.read_csv(join(path, summary_file), skiprows=3)
    summary_data.columns = summary_data.columns.str.strip()
    summary_data["idnum"] = summary_data.index
    exp_data_list = []
    for i, id_val in enumerate(summary_data["id"]):
        if i % 100 == 0:
            print(id_val)
        exp_file = join(path, f"ML2019_{species}_ML2019_{id_val}.csv")
        exp_data_list.append(pd.read_csv(exp_file))
        exp_data_list[-1]["id"] = id_val
        exp_data_list[-1]["idnum"] = i
        exp_data_list[-1]["timestep"] = np.arange(exp_data_list[-1].shape[0])
    exp_data_combined = pd.concat(exp_data_list, ignore_index=True)
    exp_data_combined.columns = exp_data_combined.columns.str.strip()
    exp_data_merged = pd.merge(exp_data_combined, summary_data, left_on="idnum", right_on="idnum")
    return exp_data_merged, summary_data
  
  exp_data_merged, summary_data = load_data("/glade/work/dgagne/Batch1_toluene", 
                                          "GAR0084_Exp_List_TEXT_PreCompile_BatchSubmit_v0.csv")
  
  #Data Summary
  
  exp_data_merged.loc[exp_data_merged["idnum"] == 10]
  
  #Input Output
  
 input_exp_cols = ['Precursor [ug/m3]',
    'Gas [ug/m3] Bin01: lg(C*) = -5.5',
    'Gas [ug/m3] Bin02: lg(C*) = -4.5',
    'Gas [ug/m3] Bin03: lg(C*) = -3.5',
    'Gas [ug/m3] Bin04: lg(C*) = -2.5',
    'Gas [ug/m3] Bin05: lg(C*) = -1.5',
    'Gas [ug/m3] Bin06: lg(C*) = -0.5',
    'Gas [ug/m3] Bin07: lg(C*) =  0.5',
    'Gas [ug/m3] Bin08: lg(C*) =  1.5',
    'Gas [ug/m3] Bin09: lg(C*) =  2.5',
    'Gas [ug/m3] Bin10: lg(C*) =  3.5',
    'Gas [ug/m3] Bin11: lg(C*) =  4.5',
    'Gas [ug/m3] Bin12: lg(C*) =  5.5',
    'Gas [ug/m3] Bin13: lg(C*) =  6.5',
    'Gas [ug/m3] Bin14: lg(C*) = -6.5',
    'Aerosol [ug_m3] Bin01: lg(C*) = -5.5',
    'Aerosol [ug_m3] Bin02: lg(C*) = -4.5',
    'Aerosol [ug_m3] Bin03: lg(C*) = -3.5',
    'Aerosol [ug_m3] Bin04: lg(C*) = -2.5',
    'Aerosol [ug_m3] Bin05: lg(C*) = -1.5',
    'Aerosol [ug_m3] Bin06: lg(C*) = -0.5',
    'Aerosol [ug_m3] Bin07: lg(C*) =  0.5',
    'Aerosol [ug_m3] Bin08: lg(C*) =  1.5',
    'Aerosol [ug_m3] Bin09: lg(C*) =  2.5',
    'Aerosol [ug_m3] Bin10: lg(C*) =  3.5',
    'Aerosol [ug_m3] Bin11: lg(C*) =  4.5',
    'Aerosol [ug_m3] Bin12: lg(C*) =  5.5',
    'Aerosol [ug_m3] Bin13: lg(C*) =  6.5',
    'Aerosol [ug_m3] Bin14: lg(C*) = -6.5',
    'temperature (K)',
    'solar zenith angle (degree)',
    'pre-existing aerosols (ug/m3)',
    'o3 (ppb)',
    'nox (ppb)',
    'oh (10^6 molec/cm3)',
    "idnum"]

output_exp_cols = ['Precursor [ug/m3]',
    'Gas [ug/m3] Bin01: lg(C*) = -5.5',
    'Gas [ug/m3] Bin02: lg(C*) = -4.5',
    'Gas [ug/m3] Bin03: lg(C*) = -3.5',
    'Gas [ug/m3] Bin04: lg(C*) = -2.5',
    'Gas [ug/m3] Bin05: lg(C*) = -1.5',
    'Gas [ug/m3] Bin06: lg(C*) = -0.5',
    'Gas [ug/m3] Bin07: lg(C*) =  0.5',
    'Gas [ug/m3] Bin08: lg(C*) =  1.5',
    'Gas [ug/m3] Bin09: lg(C*) =  2.5',
    'Gas [ug/m3] Bin10: lg(C*) =  3.5',
    'Gas [ug/m3] Bin11: lg(C*) =  4.5',
    'Gas [ug/m3] Bin12: lg(C*) =  5.5',
    'Gas [ug/m3] Bin13: lg(C*) =  6.5',
    'Gas [ug/m3] Bin14: lg(C*) = -6.5',
    'Aerosol [ug_m3] Bin01: lg(C*) = -5.5',
    'Aerosol [ug_m3] Bin02: lg(C*) = -4.5',
    'Aerosol [ug_m3] Bin03: lg(C*) = -3.5',
    'Aerosol [ug_m3] Bin04: lg(C*) = -2.5',
    'Aerosol [ug_m3] Bin05: lg(C*) = -1.5',
    'Aerosol [ug_m3] Bin06: lg(C*) = -0.5',
    'Aerosol [ug_m3] Bin07: lg(C*) =  0.5',
    'Aerosol [ug_m3] Bin08: lg(C*) =  1.5',
    'Aerosol [ug_m3] Bin09: lg(C*) =  2.5',
    'Aerosol [ug_m3] Bin10: lg(C*) =  3.5',
    'Aerosol [ug_m3] Bin11: lg(C*) =  4.5',
    'Aerosol [ug_m3] Bin12: lg(C*) =  5.5',
    'Aerosol [ug_m3] Bin13: lg(C*) =  6.5',
    'Aerosol [ug_m3] Bin14: lg(C*) = -6.5',
    "idnum", 'timestep']

input_data_list = []
output_data_list = []
for exp in np.unique(exp_data_merged.idnum):
    if exp % 100 == 0:
        print(exp)
    input_data_list.append(exp_data_merged.loc[exp_data_merged.idnum==exp, input_exp_cols].reset_index(drop=True).iloc[:-1])
    output_data_list.append(exp_data_merged.loc[exp_data_merged.idnum==exp, input_exp_cols].reset_index(drop=True).iloc[1:])
input_data = pd.concat(input_data_list, ignore_index=True)
output_data = pd.concat(output_data_list, ignore_index=True)
