import numpy as np
import pandas as pd
from os.path import join
import s3fs
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler

scalers = {"MinMaxScaler": MinMaxScaler,
           "StandardScaler": StandardScaler}

def load_data(path, aggregate_bins, species, input_columns, output_columns, log_trans_cols):
    """

    Args:
        path: Path to Data. Supports AWS S3 bucket paths.
        aggregate_bins: Weather to use aggregated data or binned (by volitility) data.
        species: Chemical species of data.
        input_columns: Input columns to use for modeling.
        output_columns: Output columns to model.

    Returns:

    """
    fs = s3fs.S3FileSystem(anon=True)
    if aggregate_bins:
        data_type = 'agg'
    else:
        data_type = 'binned'
    data = {}
    for partition in ['train_in', 'train_out', 'val_in', 'val_out']:#, 'test_in', 'test_out']:
        if 'AWS:' in path:
            aws_path = path.split('AWS:')[-1]
            data[partition] = pd.read_parquet(fs.open(join(aws_path, f'{species}_{partition}_{data_type}.parquet'))) \
                .set_index(['Time [s]', 'id'])
        else:
            data[partition] = pd.read_parquet(join(path, f'{species}_{partition}_{data_type}.parquet')) \
                .set_index(['Time [s]', 'id'])
        if '_in' in partition:
            data[partition] = data[partition][input_columns]
        elif '_out in partition':
            data[partition] = data[partition][output_columns]

        gas_log = any(["Gas" in x for x in log_trans_cols])
        aero_log = any(["Aerosol" in x for x in log_trans_cols])
        if gas_log or aero_log:
            data[partition] = data[partition].groupby('id').apply(lambda x: x.iloc[1:, :]).reset_index(level=2, drop=True)

    return data


def transform_data(data, out_path, species, tendency_cols, log_trans_cols, scaler_type, output_vars, train=False):
    """
    Transform and scale data for input into Neural network
    Args:
        data: Loaded dataframes of select species.
        tendency_cols:  Features to use tendencies instead of raw data.
        log_trans_cols: Feature to log transform.
        scaler_type: Type of sklearn scaler object
    Returns:
        numpy array of transformed and scaled data.
    """
    transformed_data = {}
    partitions = ['train_in', 'train_out', 'val_in', 'val_out']#, 'test_in', 'test_out']
    for p in partitions:

        transformed_data[p] = get_tendencies(data[p], tendency_cols)
        transformed_data[p] = log_transform(transformed_data[p], log_trans_cols)

    if train:
        x_scaler = scalers[scaler_type]()
        transformed_data['train_in'].loc[:] = x_scaler.fit_transform(transformed_data['train_in'])
        output_indices = transformed_data['train_in'].columns.get_indexer(output_vars)
        y_scaler = get_output_scaler(x_scaler, output_indices, scaler_type)

    else:
        x_scaler = joblib.load(join(out_path, 'models', f'{species}_x.scaler'))
        y_scaler = joblib.load(join(out_path, 'models', f'{species}_y.scaler'))
        transformed_data['train_in'].loc[:] = x_scaler.transform(transformed_data['train_in'])

    transformed_data['train_out'].loc[:] = y_scaler.transform(transformed_data['train_out'])
    transformed_data['val_in'].loc[:] = x_scaler.transform(transformed_data['val_in'])
    transformed_data['val_out'].loc[:] = y_scaler.transform(transformed_data['val_out'])

    return transformed_data, x_scaler, y_scaler


def get_tendencies(df, tend_cols):
    """

    Args:
        df: Pandas Dataframe including columns to get tendencies from
        tend_cols: Columns to get tendencies

    Returns:
        Pandas Dataframe with tendencies of specified columns. Removes all 'Nan' rows created.
    """
    transformed_df = df.copy(deep=True)
    transformed_df[tend_cols] = transformed_df[tend_cols].groupby('id').apply(lambda x: x[tend_cols].diff())
    transformed_df = transformed_df.dropna()
    return transformed_df


def log_transform(dataframe, cols_to_transform):
    """
    Perform log 10 transformation of specified columns
    Args:
        dataframe: full dataframe
        cols_to_transform: list of columns to perform transformation on
    """
    for col in cols_to_transform:
        if np.isin(col, dataframe.columns):
            dataframe.loc[:, col] = np.log10(dataframe[col])
    return dataframe


def log_transform_safely(dataFrame, cols_to_transform, min_value):
   """
       Performs log transform but sets any value that is negative infinty to a set min_value.
   """
   transformed = log_transform(dataFrame, cols_to_transform)
   negatives = transformed[cols_to_transform] == np.NINF
   try:
       for column in negatives.columns:
           transformed.loc[negatives[column], column] = min_value
   except:
       transformed.loc[negatives, cols_to_transform] = min_value
   return transformed


def inverse_log_transform(dataframe, cols_to_transform):
    """
    Perform inverse log 10 transformation of specified columns
    Args:
        dataframe: full dataframe
        cols_to_transform: list of columns to perform transformation on
    """

    dataframe.loc[:, cols_to_transform] = 10 ** dataframe[cols_to_transform]
    return dataframe


def inv_transform_preds(raw_preds, truth, y_scaler, log_trans_cols, tendency_cols):
    """
    Reconstruct model predictions into matching dataframe against truth
    Args:
         predictions: np.array of model predictions
         truth: pandas dataframe of truth values (including time and experiment number)
         y_scaler: outjput scaler object
         log_trans_cols: list of columns to transform from base 10 log transformations

    Returns:
         transformed pandas dataframe of predictions including time and experiment numbers
    """
    preds = raw_preds.copy()
    preds.loc[:] = y_scaler.inverse_transform(preds)
    cols = list(set(log_trans_cols) & set(raw_preds.columns))
    preds.loc[:, cols] = inverse_log_transform(preds, cols)

    if tendency_cols:
        preds_df_tend = truth.copy()
        first_time_step = truth.index.get_level_values('Time [s]').min()
        for col in tendency_cols:
            if np.isin(col, preds.columns):
                preds_df_tend.loc[preds_df_tend.index.get_level_values(
                    'Time [s]') > first_time_step, col] = preds.loc[
                    preds.index.get_level_values('Time [s]') > first_time_step, col]

        inv_tend_df = preds_df_tend.groupby('id').cumsum()
        preds = inv_tend_df.loc[inv_tend_df.index.get_level_values('Time [s]') > first_time_step]
        truth = truth.loc[truth.index.get_level_values('Time [s]') > first_time_step]

    return truth, preds


def get_output_scaler(scaler_obj, output_indices, scaler_type='MinMaxScaler'):
    """ Repopulate output scaler object with attributes from input scaler object.
    Args:
        scaler_obj: Input (x) scaler object
        output_indices: array of column indices for each output feature
        scaler_type: Sklearn scaler type from config
        data_range: data bounds for scaling from config

    Returns: output scaler
    """

    if scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler((-1, 1))
        setattr(scaler, 'scale_', scaler_obj.scale_[output_indices])
        setattr(scaler, 'min_', scaler_obj.min_[output_indices])

    elif scaler_type == 'StandardScaler':
        scaler = StandardScaler()
        setattr(scaler, 'scale_', scaler_obj.scale_[output_indices])
        setattr(scaler, 'mean_', scaler_obj.mean_[output_indices])

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
    input_scaler_df.to_csv(join(output_path, f'models/{species}_scale_values.csv'), index_label="input")
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
            # data = [y[:, 0].reshape(-1, 1), y[:, 1:].reshape(-1, 2)]
            data = [y[:, 0].reshape(-1, 1), y[:, 1:].reshape(-1, 1)]
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

