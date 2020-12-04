import sys
sys.path.append('../')
from geckoml.models import DenseNeuralNetwork, LongShortTermMemoryNetwork
from geckoml.data import combine_data, split_data, reshape_data, partition_y_output, get_output_scaler
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler, QuantileTransformer
from geckoml.metrics import ensembled_base_metrics
import tensorflow as tf
import time
import joblib
import pandas as pd
import argparse
import numpy as np
import yaml
import os
from os.path import join


def main():
    
    start = time.time()
    scalers = {"MinMaxScaler": MinMaxScaler,
               "StandardScaler": StandardScaler}
    
    # read YAML config as provided arg
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="apin_O3.yml", help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.load(config_file)

    # Extract config arguments and validate if necessary
    species = config['species']
    dir_path = config['dir_path']
    summary_file = config['summary_file']
    aggregate_bins = config['aggregate_bins']
    bin_prefix = config['bin_prefix']
    min_exp = config['min_exp']
    max_exp = config['max_exp']
    input_vars = config['input_vars']
    output_vars = config['output_vars']
    output_path = config['output_path']
    scaler_type = config['scaler_type']
    seq_length = config['seq_length']
    ensemble_members = config["ensemble_members"]
    seed = config['random_seed']
    # np.random.seed(seed)
    # tf.random.set_seed(seed)

    for folder in ['models', 'plots', 'validation_data', 'metrics', 'scalers']:
        os.makedirs(join(output_path, folder), exist_ok=True)

    # Load GECKO experiment data, split into ML inputs and outputs and persistence outputs
    input_data, output_data = combine_data(dir_path, summary_file, aggregate_bins, bin_prefix,
                                           input_vars, output_vars, min_exp, max_exp, species)

    # Split into training, validation, testing subsets
    in_train, out_train, in_val, out_val, in_test, out_test = split_data(
        input_data=input_data, output_data=output_data, random_state=seed)

    num_timesteps = in_train['Time [s]'].nunique()

    # Rescale training and validation / testing data
    x_scaler = scalers[scaler_type]((-1, 1))
    scaled_in_train = x_scaler.fit_transform(in_train.drop(['Time [s]', 'id'], axis=1))
    scaled_in_val = x_scaler.transform(in_val.drop(['Time [s]', 'id'], axis=1))

    y_scaler = get_output_scaler(x_scaler, output_vars, scaler_type, data_range=(-1, 1))
    scaled_out_train = y_scaler.transform(out_train.drop(['Time [s]', 'id'], axis=1))
    scaled_out_val = y_scaler.transform(out_val.drop(['Time [s]', 'id'], axis=1))

    val_ids = in_val['id'].values
    val_id = in_val.groupby('id').apply(lambda x: x.iloc[(seq_length - 1):, :])['id'].values

    scaled_in_train_ts, scaled_out_train_ts = reshape_data(scaled_in_train.copy(), scaled_out_train.copy(),
                                                           seq_length, num_timesteps)
    scaled_in_val_ts, scaled_out_val_ts = reshape_data(scaled_in_val.copy(), scaled_out_val.copy(),
                                                       seq_length, num_timesteps)

    # Train ML models and get validation metrics
    models, metrics = {}, {}
    for model_type in config["model_configurations"].keys():

        if model_type == 'single_ts_models':

            for model_name, model_config in config['model_configurations'][model_type].items():

                y = partition_y_output(scaled_out_train, model_config['output_layers'], aggregate_bins)

                for member in range(ensemble_members):

                    models[model_name + '_{}'.format(member)] = DenseNeuralNetwork(**model_config)
                    models[model_name + '_{}'.format(member)].fit(scaled_in_train, y)
                    preds = models[model_name + '_{}'.format(member)].predict(scaled_in_val)
                    transformed_preds = pd.DataFrame(y_scaler.inverse_transform(preds))
                    metrics[model_name + '_{}'.format(member)] = ensembled_base_metrics(
                        out_val, transformed_preds, val_ids)

        elif model_type == 'multi_ts_models':

            for model_name, model_config in config['model_configurations'][model_type].items():

                y = partition_y_output(scaled_out_train_ts, model_config['output_layers'], aggregate_bins)

                for member in range(ensemble_members):

                    models[model_name + '_{}'.format(member)] = LongShortTermMemoryNetwork(**model_config)
                    models[model_name + '_{}'.format(member)].fit(scaled_in_train_ts, y)
                    preds = models[model_name + '_{}'.format(member)].predict(scaled_in_val_ts)
                    transformed_preds = pd.DataFrame(y_scaler.inverse_transform(preds))
                    metrics[model_name + '_{}'.format(member)] = ensembled_base_metrics(
                        out_val, transformed_preds, val_id, seq_length)


    # write results
    #metrics_str = [f'{key} : {metrics[key]}' for key in metrics]
    #with open('{}metrics/{}_base_results.txt'.format(output_path, species), 'a') as f:
    #    [f.write(f'{st}\n') for st in metrics_str]
    #    f.write('\n')

    # Save ML models, scaler objects, and validation
    for model_name in models.keys():
        #models[model_name].save_fortran_model(join(output_path, 'models', model_name))
        models[model_name].model.save(join(output_path, 'models', f'{species}_{model_name}'))

    joblib.dump(x_scaler, join(output_path, 'scalers', f'{species}_x.scaler'))
    joblib.dump(y_scaler, join(output_path, 'scalers', f'{species}_y.scaler'))
    in_val.to_parquet(join(output_path, 'validation_data', f'{species}_in_val.parquet'))
    out_val.to_parquet(join(output_path, 'validation_data', f'{species}_out_val.parquet'))

    print('Completed in {0:0.1f} seconds'.format(time.time() - start))

    return


if __name__ == "__main__":
    main()
