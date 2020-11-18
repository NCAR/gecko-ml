import sys
sys.path.append('../')
from geckoml.models import DenseNeuralNetwork, LongShortTermMemoryNetwork
from geckoml.data import combine_data, split_data, reshape_data, partition_y_output
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler, QuantileTransformer
from geckoml.metrics import ensembled_base_metrics
from sklearn.pipeline import Pipeline
import tensorflow as tf
import time
import joblib
import pandas as pd
import argparse
import numpy as np
import yaml
import os

start = time.time()
seed = 8886
#np.random.seed(seed)
#tf.random.set_seed(seed)

for folder in ['models', 'plots', 'validation_data']:
    os.makedirs(os.path.join('./save_out', folder), exist_ok=True)

scalers = {"MinMaxScaler": MinMaxScaler,
           "MaxAbsScaler": MaxAbsScaler,
           "StandardScaler": StandardScaler,
           "RobustScaler": RobustScaler,
           "QuantileTransformer": QuantileTransformer}


def main():
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
    save_models = config['save_models']
    output_path = config['output_path']
    scaler_type = config['scaler_type']
    seq_length = config['seq_length']
    ensemble_members = config["ensemble_members"]

    # Load GECKO experiment data, split into ML inputs and outputs and persistence outputs
    input_data, output_data = combine_data(dir_path, summary_file, aggregate_bins, bin_prefix,
                                           input_vars, output_vars, min_exp, max_exp, species)

    # Split into training, validation, testing subsets
    in_train, out_train, in_val, out_val, in_test, out_test = split_data(
        input_data=input_data, output_data=output_data, random_state=seed)

    # Rescale training and validation / testing data
    if scaler_type == 'QuantileTransformer':
        x_scaler = Pipeline(steps=[('quant', QuantileTransformer()), ('minmax', MinMaxScaler((0, 1)))])
        y_scaler = Pipeline(steps=[('quant', QuantileTransformer()), ('minmax', MinMaxScaler((0, 1)))])
    else:

        x_scaler, y_scaler = scalers[scaler_type]((-1, 1)), scalers[scaler_type]((-1, 1))

    num_timesteps = in_train['Time [s]'].nunique()

    scaled_in_train = x_scaler.fit_transform(in_train.drop(['Time [s]', 'id'], axis=1))
    scaled_out_train = y_scaler.fit_transform(out_train.drop(['Time [s]', 'id'], axis=1))
    scaled_in_val = x_scaler.transform(in_val.drop(['Time [s]', 'id'], axis=1))
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

                for member in range(ensemble_members):

                    models[model_name + '_{}'.format(member)] = LongShortTermMemoryNetwork(**model_config)
                    models[model_name + '_{}'.format(member)].fit(scaled_in_train_ts, scaled_out_train_ts)
                    preds = models[model_name + '_{}'.format(member)].predict(scaled_in_val_ts)
                    transformed_preds = pd.DataFrame(y_scaler.inverse_transform(preds))
                    metrics[model_name + '_{}'.format(member)] = ensembled_base_metrics(
                        out_val, transformed_preds, val_id, seq_length)


    # write results
    metrics_str = [f'{key} : {metrics[key]}' for key in metrics]
    with open('{}metrics/{}_base_results.txt'.format(output_path, species), 'a') as f:
        [f.write(f'{st}\n') for st in metrics_str]
        f.write('\n')

    # Save ML models, scaler objects, and validation
    if save_models:
        for model_name in models.keys():
            #models[model_name].save_fortran_model(output_path + model_name + ".nc")
            models[model_name].model.save('{}models/{}_{}'.format(
                output_path, species, model_name))

        joblib.dump(x_scaler, '{}scalers/{}_x.scaler'.format(output_path, species))
        joblib.dump(y_scaler, '{}scalers/{}_y.scaler'.format(output_path, species))

        in_val.to_parquet('{}validation_data/{}_in_val.parquet'.format(output_path, species))
        out_val.to_parquet('{}validation_data/{}_out_val.parquet'.format(output_path, species))

    print('Completed in {0:0.1f} seconds'.format(time.time() - start))

    return


if __name__ == "__main__":
    main()
