import sys

sys.path.append('../')
import argparse
import pandas as pd
import yaml
import time
from geckoml.box import GeckoBoxEmulator
from geckoml.metrics import ensembled_metrics, save_analysis_plots, sum_bins
from geckoml.data import save_metrics, load_data, transform_data, inv_transform_preds
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
        config = yaml.safe_load(config_file)

    species = config['species']
    aggregate_bins = config['aggregate_bins']
    data_path = config['dir_path']
    output_path = config['output_path']
    exps = config['box_val_exps']
    input_vars = config['input_vars']
    output_vars = config['output_vars']
    tendency_cols = config['tendency_cols']
    log_trans_cols = config['log_trans_cols']
    ensemble_members = config['ensemble_members']
    scaler_type = config['scaler_type']
    bin_prefix = config['bin_prefix']

    data = load_data(data_path, aggregate_bins, species, input_vars, output_vars, log_trans_cols)
    transformed_data, x_scaler, y_scaler = transform_data(data, output_path, species, tendency_cols, log_trans_cols,
                                                          scaler_type, output_vars, train=False)

    metrics = {}
    for model_type in config["model_configurations"].keys():
        if model_type == 'MLP':
            for model_name in config['model_configurations'][model_type].keys():
                metrics[model_name], predictions, truth_dict = {}, {}, {}
                for member in range(ensemble_members):

                    nnet_path = join(output_path, 'models', f'{species}_{model_name}_{member}')
                    mod = GeckoBoxEmulator(neural_net_path=nnet_path,
                                           input_cols=input_vars,
                                           output_cols=output_vars)
                    raw_box_preds = mod.run_box_simulation(raw_val_output=data['val_out'],
                                                           transformed_val_input=transformed_data['val_in'],
                                                           exps=exps)

                    truth, box_preds = inv_transform_preds(raw_preds=raw_box_preds,
                                                           truth=data["val_out"],
                                                           y_scaler=y_scaler,
                                                           log_trans_cols=log_trans_cols,
                                                           tendency_cols=tendency_cols)
                    if not aggregate_bins:
                        truth, box_preds = sum_bins(truth, box_preds, bin_prefix)

                    metrics[model_name][f'member_{member}'] = ensembled_metrics(y_true=truth,
                                                                                y_pred=box_preds,
                                                                                member=member,
                                                                                output_vars=output_vars)
                    truth.loc[:, 'member'] = member
                    box_preds.loc[:, 'member'] = member
                    truth_dict[model_name + f'_{member}'] = truth
                    predictions[model_name + f'_{member}'] = box_preds

                all_truth = pd.concat(truth_dict.values())
                all_preds = pd.concat(predictions.values())
                all_preds.to_parquet(join(output_path, f'metrics/{species}_{model_name}_preds.parquet'))
                all_truth.to_parquet(join(output_path, f'metrics/{species}_{model_name}_truth.parquet'))
                save_metrics(metrics[model_name], output_path, model_name, ensemble_members, 'box')
                save_analysis_plots(all_truth, all_preds, data["train_in"], data["val_in"], output_path,
                                    output_vars, species, model_name)

        elif model_type == 'RNN':
            continue

    print('Completed in {0:0.1f} seconds'.format(time.time() - start))
    return


if __name__ == "__main__":
    main()
