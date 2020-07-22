from geckoml.models import DenseNeuralNetwork, LongShortTermMemoryNetwork
from geckoml.data import combine_data, split_data, reshape_data
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler
from geckoml.metrics import ensembled_base_metrics
import tensorflow as tf
import time
import joblib
import pandas as pd
import argparse
import numpy as np
import yaml

start = time.time()
seed = 8886
np.random.seed(seed)
tf.random.set_seed(seed)

scalers = {"MinMaxScaler": MinMaxScaler,
           "MaxAbsScaler": MaxAbsScaler,
           "StandardScaler": StandardScaler,
           "RobustScaler": RobustScaler}


def main():
    # read YAML config as provided arg
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="agg_config.yml", help="Path to config file")
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

    # Load GECKO experiment data, split into ML inputs and outputs and persistence outputs
    input_data, output_data = combine_data(dir_path, summary_file, aggregate_bins, bin_prefix,
                                           input_vars, output_vars, min_exp, max_exp, species)

    # Split into training, validation, testing subsets
    in_train, out_train, in_val, out_val, in_test, out_test = split_data(
        input_data=input_data, output_data=output_data, random_state=seed)

    #in_val.to_parquet('{}in_val_no_agg{}.parquet'.format(output_path, species))
    #out_val.to_parquet('{}out_val_no_agg{}.parquet'.format(output_path, species))
    #in_train.to_parquet('{}in_trian_no_agg{}.parquet'.format(output_path, species))
    #out_train.to_parquet('{}out_train_no_agg{}.parquet'.format(output_path, species))

    # Rescale training and validation / testing data
    x_scaler, y_scaler = scalers[scaler_type](), scalers[scaler_type]()
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
                models[model_name] = DenseNeuralNetwork(**model_config)
                models[model_name].fit(scaled_in_train, scaled_out_train)

                preds = models[model_name].predict(scaled_in_val)
                transformed_preds = pd.DataFrame(y_scaler.inverse_transform(preds))
                metrics[model_name] = ensembled_base_metrics(out_val, transformed_preds, val_ids)

        elif model_type == 'multi_ts_models':

            for model_name, model_config in config['model_configurations'][model_type].items():
                models[model_name] = LongShortTermMemoryNetwork(**model_config)
                models[model_name].fit(scaled_in_train_ts, scaled_out_train_ts)
                preds = models[model_name].predict(scaled_in_val_ts)
                transformed_preds = pd.DataFrame(y_scaler.inverse_transform(preds))
                metrics[model_name] = ensembled_base_metrics(out_val, transformed_preds, val_id, seq_length)

    for model_name, metric_values in metrics.items():
        print('{}: HD: {}, RMSE: {}'.format(model_name, metric_values[0], metric_values[1]))

    # Save ML models, scaler objects, and validation
    if save_models:
        for model_name in models.keys():

            #models[model_name].save_fortran_model(output_path + model_name + ".nc")
            models[model_name].model.save(output_path + model_name)

        joblib.dump(x_scaler, '{}{}_x.scaler'.format(output_path, species))
        joblib.dump(y_scaler, '{}{}_y.scaler'.format(output_path, species))

        in_val.to_parquet('{}in_val_{}.parquet'.format(output_path, species))
        out_val.to_parquet('{}out_val_{}.parquet'.format(output_path, species))

    print('Completed in {0:0.1f} seconds'.format(time.time() - start))

    return


if __name__ == "__main__":
    main()
