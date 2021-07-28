import sys
sys.path.append('../')
import argparse
import pandas as pd
import yaml
import time
import joblib
from geckoml.box import GeckoBoxEmulator
from geckoml.metrics import ensembled_metrics, save_analysis_plots
from geckoml.data import save_metrics
from os.path import join


def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="dodecane_agg.yml", help="Path to config file")
    parser.add_argument("-w", "--n_workers", type=int, default=50, help="Number of dask workers")
    parser.add_argument("-t", "--threads_per_worker", type=int, default=1,
                        help="Threads per dask worker (multiprocessing)")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.load(config_file)

    species = config['species']
    output_path = config['output_path']
    exps = config['box_val_exps']
    input_cols = config['input_vars']
    output_cols = config['output_vars']
    ensemble_members = config['ensemble_members']

    data = {}
    for key in ['train_in', 'train_out', 'val_in', 'val_out']:
        data[key] = pd.read_parquet(join(output_path, 'validation_data', f'{species}_{key}.parquet'))

    x_scaler = joblib.load(join(output_path, 'scalers', f'{species}_x.scaler'))
    y_scaler = joblib.load(join(output_path, 'scalers', f'{species}_y.scaler'))

    metrics = {}
    for model_type in config["model_configurations"].keys():
        if model_type == 'MLP':
            for model_name in config['model_configurations'][model_type].keys():
                metrics[model_name], predictions, truth = {}, {}, {}
                for member in range(ensemble_members):
                    nnet_path = join(output_path, 'models', f'{species}_{model_name}_{member}')
                    mod = GeckoBoxEmulator(neural_net_path=nnet_path, input_scaler=x_scaler, output_scaler=y_scaler,
                                           input_cols=input_cols, output_cols=output_cols)
                    y_true, y_preds = mod.run_box_simulation(val_data=data['val_in'], exps=exps)
                    metrics[model_name][f'member_{member}'] = ensembled_metrics(y_true, y_preds, member, output_cols)
                    y_preds['member'] = member
                    y_true['member'] = member
                    truth[model_name + f'_{member}'] = y_true
                    predictions[model_name + f'_{member}'] = y_preds

                all_preds = pd.concat(predictions.values())
                all_truth = pd.concat(truth.values())
                all_preds.to_parquet(join(output_path, f'metrics/{species}_{model_name}_preds.parquet'))
                all_truth.to_parquet(join(output_path, f'metrics/{species}_{model_name}_truth.parquet'))
                save_metrics(metrics[model_name], output_path, model_name, ensemble_members, 'box')
                save_analysis_plots(all_truth, all_preds, data["train_in"], data["val_in"], output_path,
                                    output_cols, species, model_name)

        elif model_type == 'RNN':
            continue

    print('Completed in {0:0.1f} seconds'.format(time.time() - start))
    return


if __name__ == "__main__":
    main()
