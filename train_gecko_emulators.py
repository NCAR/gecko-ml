from geckoml.models import DenseNeuralNetwork, LongShortTermMemoryNetwork
from geckoml.data import combine_data, split_data, load_combined_data, reshape_data
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler
from geckoml.metrics import ensembled_base_metrics
import tensorflow as tf
import time
import joblib
import pandas as pd
import argparse
import yaml

start = time.time()
seed = 88331
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
    save_data = config['save_data']
    output_path = config['output_path']
    scaler_type = config['scaler_type']
    seq_length = config['seq_length']

    if save_data:

        # Load GECKO experiment data, split into ML inputs and outputs and persistence outputs
        input_data, output_data = combine_data(dir_path, summary_file, aggregate_bins, bin_prefix,
                                               input_vars, output_vars, min_exp, max_exp, species)

        # Split into training, validation, testing subsets
        in_train, out_train, in_val, out_val, in_test, out_test = split_data(
            input_data=input_data, output_data=output_data, random_state=seed)

        # Save combined data to disk
        in_train.to_parquet('{}in_train_{}.parquet'.format(output_path, species))
        out_train.to_parquet('{}out_train_{}.parquet'.format(output_path, species))
        in_val.to_parquet('{}in_val_{}.parquet'.format(output_path, species))
        out_val.to_parquet('{}out_val_{}.parquet'.format(output_path, species))
        in_test.to_parquet('{}in_test_{}.parquet'.format(output_path, species))
        out_test.to_parquet('{}out_test_{}.parquet'.format(output_path, species))

    else:

        in_train, out_train, in_val, out_val, in_test, out_test = load_combined_data(output_path, species)

    # Rescale training and validation / testing data
    x_scaler, y_scaler = scalers[scaler_type](), scalers[scaler_type]()

    scaled_in_train = pd.DataFrame(x_scaler.fit_transform(in_train.drop(['Time [s]', 'id'], axis=1)))
    scaled_in_val = pd.DataFrame(x_scaler.transform(in_val.drop(['Time [s]', 'id'], axis=1)))
    scaled_out_train = pd.DataFrame(y_scaler.fit_transform(out_train.drop(['Time [s]', 'id'], axis=1)))
    scaled_out_val = pd.DataFrame(y_scaler.transform(out_val.drop(['Time [s]', 'id'], axis=1)))
    train_ids = in_train['id'].values
    val_ids = in_val['id'].values

    scaled_in_train_ts, scaled_out_train_ts = reshape_data(scaled_in_train.copy(), scaled_out_train.copy(),
                                                           seq_length, train_ids)
    scaled_in_val_ts, scaled_out_val_ts = reshape_data(scaled_in_val.copy(), scaled_out_val.copy(),
                                                       seq_length, val_ids)

    val_id = in_val.groupby('id').apply(lambda x: x.iloc[:-seq_length, :])['id'].values

    # Train ML models
    models, metrics = {}, {}
    for model_type in config["model_configurations"].keys():

        if model_type == 'single_ts_models':

            for model_name, model_config in config['model_configurations'][model_type].items():
                models[model_name] = DenseNeuralNetwork(**model_config)
                models[model_name].fit(scaled_in_train, scaled_out_train)

                seq_length = 1
                preds = models[model_name].predict(scaled_in_val)
                transformed_preds = pd.DataFrame(y_scaler.inverse_transform(preds))
                metrics[model_name] = ensembled_base_metrics(out_val, transformed_preds, val_ids, seq_length)

        elif model_type == 'multi_ts_models':

            for model_name, model_config in config['model_configurations'][model_type].items():
                models[model_name] = LongShortTermMemoryNetwork(**model_config)
                models[model_name].fit(scaled_in_train_ts, scaled_out_train_ts)

                seq_length = config['seq_length']
                preds = models[model_name].predict(scaled_in_val_ts, scaled_out_val_ts)
                transformed_preds = pd.DataFrame(y_scaler.inverse_transform(preds))
                metrics[model_name] = ensembled_base_metrics(out_val, transformed_preds, val_id, seq_length)

    for model_name, metric_values in metrics.items():
        print('{}: HD: {}, RMSE: {}'.format(model_name, metric_values[0], metric_values[1]))

    # Save ML models and scaling values to disk
    if save_models:

        for model_name in config["model_configurations"].keys():
            # models[model_name].save_fortran_model(output_path + model_name + ".nc")
            models[model_name].model.save(output_path + model_name)

        joblib.dump(x_scaler, '{}{}_X.scaler'.format(output_path, species))
        joblib.dump(y_scaler, '{}{}_Y.scaler'.format(output_path, species))

    print('Completed in {0:0.1f} seconds'.format(time.time() - start))

    return


if __name__ == "__main__":
    main()
