import sys

sys.path.append('../')
from geckoml.models import DenseNeuralNetwork
from geckoml.data import partition_y_output, inv_transform_preds, save_metrics, save_scaler_csv, load_data, \
    transform_data
from geckoml.metrics import ensembled_metrics
import time
import joblib
import argparse
import yaml
import pandas as pd
import os
from os.path import join


def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="apin_O3.yml", help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)

    species = config['species']
    data_path = config['dir_path']
    aggregate_bins = config['aggregate_bins']
    input_vars = config['input_vars']
    output_vars = config['output_vars']
    tendency_cols = config['tendency_cols']
    log_trans_cols = config['log_trans_cols']
    output_path = config['output_path']
    scaler_type = config['scaler_type']
    ensemble_members = config["ensemble_members"]
    seed = config['random_seed']

    for folder in ['models', 'plots', 'metrics']:
        os.makedirs(join(output_path, folder), exist_ok=True)

    data = load_data(data_path, aggregate_bins, species, input_vars, output_vars, log_trans_cols)
    transformed_data, x_scaler, y_scaler = transform_data(data, output_path, species, tendency_cols, log_trans_cols,
                                                          scaler_type, output_vars, train=True)

    MLP_metrics = {}
    for model_type in config["model_configurations"].keys():

        if model_type == 'MLP':

            for model_name, model_config in config['model_configurations'][model_type].items():

                y = partition_y_output(transformed_data['train_out'].values, model_config['output_layers'],
                                       aggregate_bins)
                MLP_metrics[model_name] = {}
                for member in range(ensemble_members):
                    mod = DenseNeuralNetwork(**model_config)
                    mod.fit(transformed_data['train_in'], y)
                    preds = pd.DataFrame(mod.predict(transformed_data['val_in']),
                                         columns=transformed_data['val_out'].columns,
                                         index=transformed_data['val_out'].index)
                    truth, single_ts_preds = inv_transform_preds(preds, transformed_data["val_out"], y_scaler,
                                                                 log_trans_cols, tendency_cols)
                    MLP_metrics[model_name][f'_{member}'] = ensembled_metrics(truth,
                                                                              single_ts_preds,
                                                                              member,
                                                                              output_vars)
                    print(MLP_metrics[model_name][f'_{member}'])
                    mod.model.save(join(output_path, 'models', f'{species}_{model_name}_{member}'))
                mod.save_fortran_model(join(output_path, 'models', model_name + '.nc'))
                save_metrics(MLP_metrics[model_name], output_path, model_name, ensemble_members, 'base')

        elif model_type == 'RNN':
            continue

    joblib.dump(x_scaler, join(output_path, 'models', f'{species}_x.scaler'))
    joblib.dump(y_scaler, join(output_path, 'models', f'{species}_y.scaler'))
    save_scaler_csv(x_scaler, input_vars, output_path, species, scaler_type)

    print('Completed in {0:0.1f} seconds'.format(time.time() - start))

    return


if __name__ == "__main__":
    main()
