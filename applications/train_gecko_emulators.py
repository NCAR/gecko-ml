import sys
sys.path.append('../')
from geckoml.models import DenseNeuralNetwork
from geckoml.data import partition_y_output, get_output_scaler, reconstruct_preds, save_metrics, save_scaler_csv, \
    load_data
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler, QuantileTransformer
from geckoml.metrics import ensembled_metrics
import time
import joblib
import argparse
import yaml
import os
from os.path import join


def main():
    
    start = time.time()
    scalers = {"MinMaxScaler": MinMaxScaler(),
               "StandardScaler": StandardScaler()}
    
    # read YAML config as provided arg
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="apin_O3.yml", help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.load(config_file)

    # Extract config arguments and validate if necessary
    species = config['species']
    path = config['dir_path']
    aggregate_bins = config['aggregate_bins']
    bin_prefix = config['bin_prefix']
    input_vars = config['input_vars']
    output_vars = config['output_vars']
    tendency_cols = config['tendency_cols']
    log_trans_cols = config['log_trans_cols']
    output_path = config['output_path']
    scaler_type = config['scaler_type']
    ensemble_members = config["ensemble_members"]
    seed = config['random_seed']

    for folder in ['models', 'plots', 'validation_data', 'metrics', 'scalers']:
        os.makedirs(join(output_path, folder), exist_ok=True)

    data = load_data(path, aggregate_bins, species, input_vars, output_vars, tendency_cols, log_trans_cols)

    x_scaler = scalers[scaler_type]
    scaled_in_train = x_scaler.fit_transform(data['train_in'])
    scaled_in_val = x_scaler.transform(data['val_in'])
    y_scaler = get_output_scaler(x_scaler, output_vars, scaler_type)
    scaled_out_train = y_scaler.transform(data['train_out'])

    # Train ML models and get validation metrics
    MLP_metrics = {}
    for model_type in config["model_configurations"].keys():

        if model_type == 'MLP':

            for model_name, model_config in config['model_configurations'][model_type].items():

                y = partition_y_output(scaled_out_train, model_config['output_layers'], aggregate_bins)
                MLP_metrics[model_name] = {}
                for member in range(ensemble_members):

                    mod = DenseNeuralNetwork(**model_config)
                    mod.fit(scaled_in_train, y)
                    preds = mod.predict(scaled_in_val)
                    transformed_preds = reconstruct_preds(preds, data['val_out'], y_scaler, output_vars, log_trans_cols)
                    MLP_metrics[model_name][f'_{member}'] = ensembled_metrics(data['val_out'],
                                                                              transformed_preds, member, output_vars)
                    mod.model.save(join(output_path, 'models', f'{species}_{model_name}_{member}'))
                mod.save_fortran_model(join(output_path, 'models', model_name + '.nc'))
                save_metrics(MLP_metrics[model_name], output_path, model_name, ensemble_members, 'base')

        elif model_type == 'RNN':
            continue

    joblib.dump(x_scaler, join(output_path, 'scalers', f'{species}_x.scaler'))
    joblib.dump(y_scaler, join(output_path, 'scalers', f'{species}_y.scaler'))
    save_scaler_csv(x_scaler, input_vars, output_path, species, scaler_type)

    for key in data.keys():
        data[key].to_parquet(join(output_path, 'validation_data', f'{species}_{key}.parquet'))

    print('Completed in {0:0.1f} seconds'.format(time.time() - start))

    return


if __name__ == "__main__":
    main()
