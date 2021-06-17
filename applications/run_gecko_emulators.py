import sys
sys.path.append('../')
import argparse
import pandas as pd
import yaml
import time
import joblib
from geckoml.box import GeckoBoxEmulator, GeckoBoxEmulatorTS
from geckoml.metrics import ensembled_metrics, match_true_exps, plot_ensemble, plot_bootstrap_ci, \
    plot_crps_bootstrap, plot_unstability, plot_scatter_analysis
from geckoml.data import inverse_log_transform, save_metrics
from dask.distributed import Client, LocalCluster
from os.path import join


def main():
    start = time.time()
    # read YAML config as provided arg
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="dodecane_agg.yml", help="Path to config file")
    parser.add_argument("-w", "--n_workers", type=int, default=50, help="Number of dask workers")
    parser.add_argument("-t", "--threads_per_worker", type=int, default=1,
                        help="Threads per dask worker (multiprocessing)")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.load(config_file)

    # Extract config arguments and validate if necessary
    species = config['species']
    output_path = config['output_path']
    exps = config['box_val_exps']
    input_cols = config['input_vars']
    output_cols = config['output_vars']
    ensemble_members = config['ensemble_members']
    columns = ['Precursor [ug/m3]', 'Gas [ug/m3]', 'Aerosol [ug_m3]']

    # Read validation data and scaler objects
    train_in = pd.read_parquet(join(output_path, 'validation_data', f'{species}_in_train.parquet'))
    val_in = pd.read_parquet(join(output_path, 'validation_data', f'{species}_in_val.parquet'))
    val_out = pd.read_parquet(join(output_path, 'validation_data', f'{species}_out_val.parquet'))
    val_out = inverse_log_transform(val_out, ['Precursor [ug/m3]'])

    x_scaler = joblib.load(join(output_path, 'scalers', f'{species}_x.scaler'))
    y_scaler = joblib.load(join(output_path, 'scalers', f'{species}_y.scaler'))

    scaled_val_arr = x_scaler.transform(val_in.iloc[:, 1:-1])
    scaled_val_in = val_in.copy()
    scaled_val_in[input_cols[1:-1]] = scaled_val_arr
    n_time_steps = scaled_val_in['Time [s]'].nunique()

    # Run multiple GECKO experiments in parallel
    cluster = LocalCluster(processes=True, n_workers=args.n_workers, threads_per_worker=args.threads_per_worker)
    client = Client(cluster)
    single_ts_metrics, multi_ts_metrics = {}, {}
    for model_type in config["model_configurations"].keys():
        if model_type == 'single_ts_models':
            for model_name in config['model_configurations'][model_type].keys():
                seq_length = 1
                single_ts_metrics[model_name], predictions, truth = {}, {}, {}
                for member in range(ensemble_members):
                    nnet_path = join(output_path, 'models', f'{species}_{model_name}_{member}')
                    mod = GeckoBoxEmulator(neural_net_path=nnet_path, output_scaler=y_scaler,
                                           input_cols=input_cols, output_cols=output_cols, seed=config['random_seed'])

                    box_preds = mod.run_ensemble(client=client, data=scaled_val_in, num_timesteps=n_time_steps,
                                                 exps=exps)
                    y_true, y_preds = match_true_exps(truth=val_out, preds=box_preds, num_timesteps=n_time_steps,
                                                      seq_length=seq_length, aggregate_bins=config['aggregate_bins'],
                                                      bin_prefix=config['bin_prefix'])

                    single_ts_metrics[model_name][f'member_{member}'] = ensembled_metrics(y_true, y_preds, member)
                    y_preds['member'] = member
                    y_true['member'] = member
                    truth[model_name + f'_{member}'] = y_true
                    predictions[model_name + f'_{member}'] = y_preds

                plot_ensemble(truth=y_true, preds=predictions, output_path=output_path,
                              species=species, model_name=model_name)
                all_preds = pd.concat(predictions.values())
                all_truth = pd.concat(truth.values())
                all_preds.to_csv(join(output_path, f'metrics/{species}_{model_name}_preds.csv'), index=False)
                all_truth.to_csv(join(output_path, f'metrics/{species}_{model_name}_truth.csv'), index=False)
                save_metrics(single_ts_metrics[model_name], output_path, model_name, ensemble_members, 'box')
                plot_bootstrap_ci(all_truth, all_preds, columns, output_path, species, model_name)
                plot_crps_bootstrap(all_truth, all_preds, columns, output_path, species, model_name)
                plot_unstability(all_preds, columns, output_path, model_name)
                plot_scatter_analysis(all_preds, all_truth, train_in, val_in, columns[1:],
                                      output_path, species, model_name)

        elif model_type == 'multi_ts_models':
            for model_name in config['model_configurations'][model_type].keys():
                seq_length = config['seq_length']
                multi_ts_metrics[model_name], predictions = {}, {}
                for member in range(ensemble_members):
                    nnet_path = join(output_path, 'models', f'{species}_{model_name}_{member}')
                    mod = GeckoBoxEmulatorTS(neural_net_path=nnet_path, output_scaler=y_scaler, seq_length=seq_length,
                                                 input_cols=input_cols, output_cols=output_cols, seed=config['random_seed'])
                    box_preds = mod.run_ensemble(client=client, data=scaled_val_in, num_timesteps=n_time_steps,
                                                 exps=exps)
                    y_true, y_preds = match_true_exps(truth=val_out, preds=box_preds, num_timesteps=n_time_steps,
                                                 seq_length=seq_length, aggregate_bins=config['aggregate_bins'],
                                                 bin_prefix=config['bin_prefix'])
                    multi_ts_metrics[model_name][f'_{member}'] = ensembled_metrics(y_true, y_preds, member)
                    predictions[model_name + f'_{member}'] = y_preds

                plot_ensemble(truth=y_true, preds=predictions, output_path=output_path,
                              species=species, model_name=model_name)
                plot_scatter_analysis(all_preds, all_truth, train_in, val_in, ['Gas [ug/m3', 'Aersol [ug_m3'],
                                      output_path, species, model_name)
                save_metrics(multi_ts_metrics[model_name], output_path, model_name, ensemble_members, 'box')

    print('Completed in {0:0.1f} minutes.'.format((time.time() - start) / 60))
    return


if __name__ == "__main__":
    main()
