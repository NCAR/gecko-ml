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
           "StandardScaler": StandardScaler}


def add_diurnal_signal(x_data):
    """
    Apply Function to static temperature to add +- 4 [K] diurnal signal (dependent of time [s] of timeseries).
    Args:
        x_data: Pre-scaled/normalized input data (Pandas DF).

    Returns: Same df with function applied to temperature feature.
    """
    x_data['temperature (K)'] = x_data['temperature (K)'] + 4.0 * np.sin(
        (x_data['Time [s]'] * 7.2722e-5 + (np.pi / 2.0 - 7.2722e-5 * 64800.0)))

    return x_data


def log_transform(dataframe, cols_to_transform):
    """
    Perform log 10 transformation of specified columns
    Args:
        dataframe: full dataframe
        cols_to_transform: list of columns to perform transformation on 
    """
    dataframe[cols_to_transform] = np.log10(dataframe[cols_to_transform])
    return dataframe


def inverse_log_transform(dataframe, cols_to_transform):
    """
    Perform inverse log 10 transformation of specified columns
    Args:
        dataframe: full dataframe
        cols_to_transform: list of columns to perform transformation on 
    """
    dataframe[cols_to_transform] = 10 ** dataframe[cols_to_transform]
    return dataframe


def reconstruct_preds(predictions, truth, y_scaler, log_trans_cols, seq_len=1):
    """
    Reconstruct model predictions into matching dataframe against truth
    Args:
         predictions: np.array of model predictions
         truth: pandas dataframe of truth values (including time and experiment number)
         y_scaler: outjput scaler object
         log_trans_cols: list of columns to transform from base 10 log transformations
         seq_len: lookback length for lstm models (otherwise defaults to 1)
    Returns:
         transformed pandas dataframe of predictions including time and experiment numbers
    """
    preds = pd.DataFrame(y_scaler.inverse_transform(predictions), columns=truth.columns[1:-1])
    truth = truth.groupby('id').apply(lambda x: x.iloc[seq_len - 1:, :]).reset_index(drop=True)
    preds['Time [s]'] = truth['Time [s]'].values
    preds['id'] = truth['id'].values
    preds = inverse_log_transform(preds, log_trans_cols)
    preds = preds.sort_values(['id', 'Time [s]']).reset_index(drop=True)

    return preds

def get_tendencies(df, output_cols):
    """
     Transform dataframe to time tendencies rather than actual values. Preserves static environmental values.
    Args:
        df: Pre-scaled input dataframe.
        output_cols: Output columns to be transformed (should include 'id' and 'Time' for indexing).

    Returns: Pandas dataframe with input columns transformed to tendencies (Removes the first sample of each Exp).
    """
    df_copy = df.copy()
    dummy_df = df[output_cols].drop(['Time [s]'], axis=1).groupby('id').diff().reset_index(drop=True)
    df_copy[output_cols[1:-1]] = dummy_df.values
    df_copy.loc[:, ~df_copy.columns.isin(output_cols)] = df.loc[:, ~df.columns.isin(output_cols)]
    dff = df_copy.groupby('id').apply(lambda x: x.iloc[1:, :]).reset_index(drop=True)

    return dff


def convert_to_values(original, preds, output_cols, seq_length=1):
    """
    Convert tendencies back to actual values.
    Args:
        original: Original df that was used to create tendencies
        preds: Model predictions.
        output_cols: Output columns from config.
        seq_length: Length of sequence (for multi-time step models)
    Returns: Converted Dataframe.
    """
    original = original[output_cols].groupby('id').apply(lambda x: x.iloc[seq_length - 1:-1, :]).reset_index(drop=True)
    original = original.loc[:, original.columns.isin(output_cols)].drop(['Time [s]', 'id'], axis=1).reset_index(
        drop=True)
    preds.columns = output_cols[1:-1]
    converted = original.add(preds)[output_cols[1:-1]]

    return converted


def get_data_serial(file_path, summary_data, bin_prefix, input_vars, output_vars, aggregate_bins):
    """
        Load an experiment file based on a summary file; combine data from summary into experiment file
    
    Args:
        file_path: Experiment file to load
        summary_data: Summary dataframe
        bin_prefix: Prefix of compound volitility bins if aggregation is used
        input_vars: List of varibles to subset for input
        output_vars: List of varibles to subset for ouput
        aggregate_bins: Boolean to aggregate bins
    
    Returns:
        input_subset: All input variable time series data for every experiment in pandas DataFrame
        output_subset: All ouput variable time series data for every experiment in pandas DataFrame
                       lagged by one time step
    """
    
    df = pd.read_csv(file_path)
    df.columns = [x.strip() for x in df.columns]
    summary_data.columns = [x.strip() for x in summary_data.columns]
    
    exp_num = int(re.findall("_Exp(\d+)*", file_path)[0])
    
    for variable in summary_data.columns:
        
        df[variable] = summary_data[summary_data['id'] == 
                                    'Exp{}'.format(exp_num)][variable][exp_num]
    if aggregate_bins: 
        
        for prefix in bin_prefix:

            df[prefix] = df.loc[:, df.columns.str.contains(prefix, regex=False)].sum(axis=1)
            
    input_subset, output_subset = df[input_vars].iloc[:-1, :], df[output_vars].iloc[1:, :]
    
    return input_subset, output_subset


def combine_data(dir_path, summary_file, aggregate_bins, bin_prefix,
                 input_vars, output_vars, species):
    """
        Distribute get_serial_data() using dask to parallelize tasks and concatenate dataframes
    Args:
        dir_path: Directory path for experiment files
        summary_file: Name of summary file
        aggregate_bins (boolean): Whether to sum data of similar type
        bin_prefix: List of prefixes fr aggregating (if aggregate_bins is True)
        input_vars: List of input features
        output_vars: List of output features
        min_exp: Minimum experiment number to process
        max_exp: Maximum experiment number to process
        species: Chemical species from config
    
    Returns:
        input_subset: Complete pandas input dataframe 
        output_subset: Complete pandas output dataframe (lagged by one time step)
    """
    
    file_list = glob.glob('{}ML2019_{}_ML2019_Exp*'.format(dir_path, species))
    sorted_filelist = sorted(file_list, key=lambda x:list(map(int, re.findall("_Exp(\d+)*", x))))
    
    summary = pd.read_csv(dir_path+summary_file, skiprows=3)
    summary.columns = [x.strip() for x in summary.columns]
    
    dfs_in = [delayed(get_data_serial)
              (f, summary, bin_prefix=bin_prefix, input_vars=input_vars, output_vars=output_vars,
               aggregate_bins=aggregate_bins)[0] for f in sorted_filelist] 
    dfs_out = [delayed(get_data_serial)
               (f, summary, bin_prefix=bin_prefix, input_vars=input_vars, output_vars=output_vars,
                aggregate_bins=aggregate_bins)[1] for f in sorted_filelist] 

    ddf_in, ddf_out = dd.from_delayed(dfs_in), dd.from_delayed(dfs_out)  # assemble dask dfs
    
    df_in = ddf_in.compute(scheduler='processes').reset_index(drop=True) # transform back to pandas df
    df_out = ddf_out.compute(scheduler='processes').reset_index(drop=True)
    df_in = log_transform(df_in, ['Precursor [ug/m3]'])
    df_in = add_diurnal_signal(df_in)
    df_out = log_transform(df_out, ['Precursor [ug/m3]'])

    return df_in, df_out


def get_output_scaler(scaler_obj, output_vars, scaler_type='MinMaxScaler', data_range=(-1, 1)):
    """ Repopulate output scaler object with attributes from input scaler object.
    Args:
        scaler_obj: Input (x) scaler object
        output_vars: list of output variables from config
        scaler_type: Sklearn scaler type from config
        data_range: data bounds for scaling from config

    Returns: output scaler
    """
    num_features = len(output_vars[1:-1])
    if scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler(data_range)
        setattr(scaler, 'scale_', scaler_obj.scale_[0:num_features])
        setattr(scaler, 'min_', scaler_obj.min_[0:num_features])

    elif scaler_type == 'StandardScaler':
        scaler = StandardScaler(data_range)
        setattr(scaler, 'scale_', scaler_obj.scale_[0:num_features])
        setattr(scaler, 'mean_', scaler_obj.mean_[0:num_features])

    return scaler


def partition_y_output(y, output_layers, aggregate_bins=True):
    """
    Split y data into list based on number of output layers
    Args:
        y: scaled y data (np.array)
        output_layers: number of output layer from config file
        aggregate_bins (boolean): Whether data was aggregated (determines number of features)

    Returns: list of y data to be fed to fit function
    """
    if (output_layers > 3) | (output_layers < 1):
        raise ValueError('Invalid number of layers. Must be either 1, 2 or 3.')
    elif output_layers == 3:
        if not aggregate_bins:
            data = [y[:, 0].reshape(-1, 1), y[:, 1:15].reshape(-1, 14), y[:, 15:].reshape(-1, 14)]
        else:
            data = [y[:, 0].reshape(-1, 1), y[:, 1].reshape(-1, 1), y[:, 2].reshape(-1, 1)]
    elif output_layers == 2:
        if not aggregate_bins:
            data = [y[:, 0].reshape(-1, 1), y[:, 1:].reshape(-1, 28)]
        else:
            data = [y[:, 0].reshape(-1, 1), y[:, 1:].reshape(-1, 2)]
    elif output_layers == 1:
        data = [y]
    return data


def split_data(input_data, output_data, train_start, train_end, val_start, val_end, test_start, test_end):
    """
    Split data, by experiment, into training/validation/testing sets for both input/output dataframes.
    Args:
        input_data: Complete input dataframe to process
        output_data: Complete output dataframe to process
        train_start (int): Experiment number to start from for training set
        train_end (int): Experiment number to end on for training set
        val_start (int): Experiment number to start from for validation set
        val_end (int): Experiment number to end on for validation set
        test_start (int): Experiment number to start from for test set
        test_end (int): Experiment number to end on for test set
    Returns:
        in_train: training data input as pandas df (80% total input)
        out_train: training data output as pandas df (80% total output)
        in_val: validation data input as pandas df (10% total input)
        out_val: validation data output as pandas df (10% total output)
        in_test: testing data input as pandas df (10% total input)
        out_test: testing data output as pandas df (10% total output)
    """
    train_exps = np.arange(train_start, train_end + 1)
    val_exps = np.arange(val_start, val_end + 1)
    test_exps = np.arange(test_start, test_end + 1)

    in_train = input_data[input_data['id'].isin(["".join(('Exp', str(i))) for i in train_exps])]
    out_train = output_data[output_data['id'].isin(["".join(('Exp', str(i))) for i in train_exps])]
    in_val = input_data[input_data['id'].isin(["".join(('Exp', str(i))) for i in val_exps])]
    out_val = output_data[output_data['id'].isin(["".join(('Exp', str(i))) for i in val_exps])]
    in_test = input_data[input_data['id'].isin(["".join(('Exp', str(i))) for i in test_exps])]
    out_test = output_data[output_data['id'].isin(["".join(('Exp', str(i))) for i in test_exps])]

    return in_train, out_train, in_val, out_val, in_test, out_test


def reshape_data(x_data, y_data, seq_length, num_timesteps):
    """
    Reshape matrix data into sample shape for LSTM training.

    Args:
        x_data: DataFrame containing input features (columns) and time steps (rows).
        y_data: Matrix containing output features (columns) and time steps (rows).
        seq_length: Length of look back time steps for one time step of prediction.
        num_timesteps: Number of time_steps per experiment.

    Returns: Two np.ndarrays, the first of shape (samples, length of sequence,
        number of features), containing the input data for the LSTM. The second
        of shape (samples, number of output features) containing the expected output for each input
        sample.
    """
    num_samples, num_features = x_data.shape
    num_output = y_data.shape[1]
    num_exps = int(num_samples / num_timesteps)
    num_seq_ts = num_timesteps - seq_length + 1

    x_new = np.zeros((num_exps * num_seq_ts, seq_length, num_features))
    y_new = np.zeros((num_exps * num_seq_ts, num_output))
    for i in range(num_exps):
        for j in range(num_seq_ts):
            x_new[(i * num_seq_ts) + j] = x_data[(num_timesteps * i) + j:(num_timesteps * i) + j + seq_length]
            y_new[(i * num_seq_ts) + j] = y_data[(num_timesteps * i) + j + seq_length - 1]

    return x_new, y_new


def save_metrics(metrics, output_path, model_name, members, model_type):
    """
    Save ensemble member metrics and aggregated metrics to CSV file 
    Args:
        metrics: dictionary of pandas dataframes of metrics for each ensemble member
        output_path: Base output path 
        model_name: name of model used to create metrics
        members: number of ensemble members
        model_type: (str) 'base' or 'box'. Used for file naming. 
    """
    member_metrics = pd.concat([x for x in metrics.values()]).reset_index(drop=True)
    ensemble_metrics = member_metrics.groupby(['mass_phase']).mean()
    ensemble_metrics['n_members'] = members
    member_metrics.to_csv(join(output_path, 'metrics', f'{model_name}_{model_type}_metrics.csv'), index=False)
    ensemble_metrics.to_csv(join(output_path, 'metrics', f'{model_name}_mean_{model_type}_metrics.csv'))

    return

