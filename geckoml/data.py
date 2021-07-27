import numpy as np
import pandas as pd
from os.path import join
import s3fs
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler

scalers = {"MinMaxScaler": MinMaxScaler,
           "StandardScaler": StandardScaler}

def load_data(path, aggregate_bins, species, input_columns, output_columns):
    """

    Args:
        path:
        aggregate_bins:
        species:
        input_columns:
        output_columns:

    Returns:

    """
    fs = s3fs.S3FileSystem(anon=True)
    if aggregate_bins:
        data_type = 'agg'
    else:
        data_type = 'binned'
    data = {}
    for partition in ['train_in', 'train_out', 'val_in', 'val_out']:
        data[partition] = pd.read_parquet(fs.open(join(path, f'{species}_{partition}_{data_type}.parquet'))) \
            .set_index(['Time [s]', 'id'])
        if '_in' in partition:
            data[partition] = data[partition][input_columns]
        elif '_out in partition':
            data[partition] = data[partition][output_columns]
    return data

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
    dataframe.loc[:, cols_to_transform] = 10 ** dataframe[cols_to_transform]
    return dataframe


def reconstruct_preds(predictions, truth, y_scaler, output_columns, log_trans_cols):
    """
    Reconstruct model predictions into matching dataframe against truth
    Args:
         predictions: np.array of model predictions
         truth: pandas dataframe of truth values (including time and experiment number)
         y_scaler: outjput scaler object
         seq_len: list of output columns of model
         log_trans_cols: list of columns to transform from base 10 log transformations

    Returns:
         transformed pandas dataframe of predictions including time and experiment numbers
    """
    preds = truth.copy(deep=True)
    preds.loc[:, output_columns] = y_scaler.inverse_transform(predictions)
    if not log_trans_cols:
        preds.loc[:, log_trans_cols] = inverse_log_transform(preds, log_trans_cols)

    return preds


def get_output_scaler(scaler_obj, output_vars, scaler_type='MinMaxScaler', data_range=(-1, 1)):
    """ Repopulate output scaler object with attributes from input scaler object.
    Args:
        scaler_obj: Input (x) scaler object
        output_vars: list of output variables from config
        scaler_type: Sklearn scaler type from config
        data_range: data bounds for scaling from config

    Returns: output scaler
    """
    num_features = len(output_vars)
    if scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler(data_range)
        setattr(scaler, 'scale_', scaler_obj.scale_[0:num_features])
        setattr(scaler, 'min_', scaler_obj.min_[0:num_features])

    elif scaler_type == 'StandardScaler':
        scaler = StandardScaler(data_range)
        setattr(scaler, 'scale_', scaler_obj.scale_[0:num_features])
        setattr(scaler, 'mean_', scaler_obj.mean_[0:num_features])

    return scaler


def save_scaler_csv(scaler_obj, input_columns, output_path, species, scaler_type='StandardScaler'):
    """
    Save the scaler information to csv so that it can be read later.
    Args:
        scaler_obj: Scikit-learn StandardScaler object
        input_columns: List of scaled columns
        output_path: Path to dave file to
        scaler_type: Type of scaler object. Supports 'MinMaxScaler' and 'StandardScaler'
    Returns:
    """
    if scaler_type == 'StandardScaler':
        input_scaler_df = pd.DataFrame({"mean": scaler_obj.mean_, "scale": scaler_obj.scale_},
                                       index=input_columns)
    elif scaler_type == 'MinMaxScaler':
        input_scaler_df = pd.DataFrame({"min": scaler_obj.min_, "scale": scaler_obj.scale_},
                                       index=input_columns)
    input_scaler_df.to_csv(join(output_path, f'scalers/{species}_scale_values.csv'), index_label="input")
    return


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
    ensemble_metrics = member_metrics.groupby(['mass_phase']).mean().sort_values(
        by='mass_phase')
    ensemble_metrics['n_members'] = members
    member_metrics.to_csv(join(output_path, 'metrics', f'{model_name}_{model_type}_metrics.csv'), index=False)
    ensemble_metrics.to_csv(join(output_path, 'metrics', f'{model_name}_mean_{model_type}_metrics.csv'))

    return

