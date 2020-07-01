import numpy as np
import pandas as pd
from os.path import join, exists
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm
import glob
import re
import dask.dataframe as dd
from dask import delayed

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

def get_data_serial(file_path, summary_data, bin_prefix, input_vars, output_vars, aggregate_bins):
    """
        Load an experiment file based on a summary file; combine data from summary into experiment file
    
    Args:
        file_path: Experiment file to load
        summary_data: Summary dataframe
        bin_prefix: Prefix of compound volitility bins if aggregation is used
        input_vars: List of varibles to subset for input
        ouput_vars: List of varibles to subset for ouput
        aggregate_bins: Boolean to aggregate bins
    
    Returns:
        input_subset: All input variable time series data for every experiment in pandas DataFrame
        output_subset: All ouput variable time series data for every experiment in pandas DataFrame
                       lagged by one time step
    """
    
    df = pd.read_csv(file_path)
    df.columns = [x.strip() for x in df.columns]
    summary_data.columns = [x.strip() for x in summary_data.columns]
    
    exp_num = int(re.findall("_Exp(\d+).csv", file_path)[0])
    
    for variable in summary_data.columns:
        
        df[variable] = summary_data[summary_data['id'] == 
                                    'Exp{}'.format(exp_num)][variable][exp_num]
    if aggregate_bins: 
        
        for prefix in bin_prefix:

            df[prefix] = df.loc[:, df.columns.str.contains(prefix, regex=False)].sum(axis=1)
            
    input_subset, output_subset = df[input_vars].iloc[:-1,:], df[output_vars].iloc[1:,:]
    
    return input_subset, output_subset


def combine_data(dir_path, summary_file, aggregate_bins, bin_prefix,
                 input_vars, output_vars, min_exp, max_exp, species):
    """
        Distribute get_serial_data() using dask to parallelize tasks and concatenate dataframes 
    
    Args:
        summary_path: Full path of summary file
        min_exp: Minimum experiment number to process
        max_exp: Maximum experiment number to process
    
    Returns:
        input_subset: Complete pandas input dataframe 
        output_subset: Complete pandas output dataframe (lagged by one time step)
    """
    
    file_list = glob.glob('{}ML2019_{}_ML2019_Exp*'.format(dir_path, species))
    sorted_filelist = sorted(file_list, key=lambda x:list(map(int, re.findall("_Exp(\d+).csv", x))))
    
    summary = pd.read_csv(dir_path+summary_file, skiprows=3)
    summary.columns = [x.strip() for x in summary.columns]
    
    dfs_in = [delayed(get_data_serial)
              (f, summary, bin_prefix=bin_prefix, input_vars=input_vars, output_vars=output_vars,
               aggregate_bins=aggregate_bins)[0] for f in sorted_filelist[min_exp:max_exp + 1]] 
    dfs_out = [delayed(get_data_serial)
               (f, summary, bin_prefix=bin_prefix, input_vars=input_vars, output_vars=output_vars,
                aggregate_bins=aggregate_bins)[1] for f in sorted_filelist[min_exp:max_exp + 1]] 

    ddf_in, ddf_out = dd.from_delayed(dfs_in), dd.from_delayed(dfs_out)  # assemble dask dfs
    
    df_in = ddf_in.compute(scheduler='processes').reset_index() # transform back to pandas df
    df_out = ddf_out.compute(scheduler='processes').reset_index() 

    del df_in['index'], df_out['index']
        
    return df_in, df_out

def split_data(input_data, output_data, n_splits=2, random_state=8):
    """
        Split data, by experiment, into training/validation/testing sets for both input/output dataframes
    
    Args:
        input_data: Complete input dataframe to process
        output_data: Complete output dataframe to process
        n_splits: Number of re-shuffling & splitting iterations
    
    Returns:
        in_train: training data input as pandas df (80% total input)
        in_val: validation data input as pandas df (10% total input)
        in_test: testing data input as pandas df (10% total input)
        out_train: training data output as pandas df (80% total output)
        out_val: validation data output as pandas df (10% total output)
        out_test: testing data output as pandas df (10% total output)
    """
    train_indx, remain_indx = next(GroupShuffleSplit(test_size=.2, n_splits=n_splits,
                                    random_state=random_state).split(input_data, groups=input_data['id']))
    
    in_train, out_train = input_data.iloc[train_indx], output_data.iloc[train_indx]
    remain_in, remain_out = input_data.iloc[remain_indx], output_data.iloc[remain_indx]
    
    val_indx, test_indx = next(GroupShuffleSplit(test_size=.5, n_splits=n_splits,
                                    random_state=random_state).split(remain_in, groups=remain_in['id']))
     
    in_val, out_val = remain_in.iloc[val_indx], remain_out.iloc[val_indx]
    in_test, out_test = remain_in.iloc[test_indx], remain_out.iloc[test_indx]

    return in_train, out_train, in_val, out_val, in_test, out_test

def get_starting_conds(dir_path, summary_file, bin_prefix, input_vars, output_vars,
                       aggregate_bins, species, exp_num, starting_ts=0):
    """
    Get starting conditions for Box emulator.
    Args:
        dir_path: Base directory of experiment/summary data
        summary_file: Full path of the summary file 
        bin_prefix: Prefix of compound volitility bins if aggregation is used
        input_vars: List of varibles to subset for input
        ouput_vars: List of varibles to subset for ouput
        aggregate_bins: Boolean to aggregate bins
        species: Which molecular species
        exp_num: Which experiment the box model will be run on
        starting_ts: Starting timestep to use for box emulator
    Returns:
        ts_data: Timestep data to be used as initial input to box emulator (np.array)
        
    """
    
    file_name = '{}ML2019_{}_ML2019_Exp{}.csv'.format(dir_path, species, exp_num)
    summary = pd.read_csv(dir_path+summary_file, skiprows=3) 
    exp_data = get_data_serial(file_name, summary, bin_prefix, input_vars, output_vars, aggregate_bins)[0]
    
    ts_data = exp_data.iloc[starting_ts:starting_ts+1, :-1].values

    return ts_data

def repopulate_scaler(scale_data, scaler_type="MinMaxScaler"):
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
    for column in scale_data.columns[1:]:
        setattr(scaler_obj, column + "_", scale_data[column].values)
    return scaler_obj
