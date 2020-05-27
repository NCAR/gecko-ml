import numpy as np
import pandas as pd
from os.path import join, exists
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler
from tqdm import tqdm

scalers = {"MinMaxScaler": MinMaxScaler,
           "MaxAbsScaler": MaxAbsScaler,
           "StandardScaler": StandardScaler,
           "RobustScaler": RobustScaler}

input_exp_cols_default = ['Precursor [ug/m3]',
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

output_exp_cols_default = ['Precursor [ug/m3]',
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


def load_data(path, summary_file, species="toluene_kOH", delimiter=", ", experiment="ML2019"):
    """
    Load a set of experiment files based on a summary file.
    
    Args:
        path: Path to the directory containing summary and experiment files
        summary_file: Name of the summary file (should not contain the path)
        species: Name of the precursor chemical species
        delimiter
    
    Returns:
        exp_data_merged: All time series data for every experiment in pandas DataFrame
        summary_data: Summary experiment data file in pandas DataFrame
    """
    summary_data = pd.read_csv(join(path, summary_file), skiprows=3)
    summary_data.columns = summary_data.columns.str.strip()
    summary_data["idnum"] = summary_data.index
    exp_data_list = []
    for i, id_val in enumerate(tqdm(summary_data["id"])):
        exp_file = join(path, f"{experiment}_{species}_{experiment}_{id_val}.csv")
        exp_data_list.append(pd.read_csv(exp_file, delimiter=delimiter))
        exp_data_list[-1]["id"] = id_val
        exp_data_list[-1]["idnum"] = int(id_val[3:])
        exp_data_list[-1]["timestep"] = np.arange(exp_data_list[-1].shape[0])
    exp_data_combined = pd.concat(exp_data_list, ignore_index=True)
    exp_data_combined.columns = exp_data_combined.columns.str.strip()
    exp_data_merged = pd.merge(exp_data_combined, summary_data, left_on="idnum", right_on="idnum")
    return exp_data_merged, summary_data
  
#exp_data_merged, summary_data = load_data("/glade/work/dgagne/Batch1_toluene",
#                                          "GAR0084_Exp_List_TEXT_PreCompile_BatchSubmit_v0.csv")


def extract_inputs_outputs(exp_data_merged, input_exp_cols, output_exp_cols):
    input_data_list = []
    output_data_list = []
    for exp in np.unique(exp_data_merged.idnum):
        if exp % 100 == 0:
            print(exp)
        input_data_list.append(
            exp_data_merged.loc[exp_data_merged.idnum == exp, input_exp_cols].reset_index(drop=True).iloc[:-1])
        output_data_list.append(
            exp_data_merged.loc[exp_data_merged.idnum == exp, output_exp_cols].reset_index(drop=True).iloc[1:])
    input_data = pd.concat(input_data_list, ignore_index=True)
    output_data = pd.concat(output_data_list, ignore_index=True)
    return input_data, output_data


def repopulate_scaler(scale_file, scaler_type="StandardScaler"):
    """
    Given a csv file containing scaling values, repopulate a scikit-learn
    scaling object with those same values.

    Args:
        scale_file (str): path to csv file containing scale values
        scaler_type (str):

    Returns:
        scaler_obj: Scaler Object with normalizing values specified.
    """
    scaler_obj = scalers[scaler_type]()
    scale_data = pd.read_csv(scale_file)
    for column in scale_data.columns[1:]:
        setattr(scaler_obj, column + "_", scale_data[column].values)
    return scaler_obj

