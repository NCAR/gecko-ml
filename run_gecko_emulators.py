import argparse
import pandas as pd
import yaml
import time
import joblib
import tensorflow as tf
from geckoml.box import GeckoBoxEmulator, GeckoBoxEmulatorTS
from geckoml.metrics import ensembled_box_metrics, plot_mae_ts, match_true_exps, plot_ensemble
from dask.distributed import Client, LocalCluster


start = time.time()


def main():
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
    num_exps = config['num_exps']
    input_cols = config['input_vars']
    output_cols = config['output_vars']
    ensemble_members = config['ensemble_members']

    # Read validation data and scaler objects
    val_in = pd.read_parquet('{}validation_data/{}_in_val.parquet'.format(output_path, species))
    val_out = pd.read_parquet('{}validation_data/{}_out_val.parquet'.format(output_path, species))

    x_scaler = joblib.load('{}scalers/{}_x.scaler'.format(output_path, species))
    y_scaler = joblib.load('{}scalers/{}_y.scaler'.format(output_path, species))

    scaled_val_arr = x_scaler.transform(val_in.iloc[:, 1:-1])
    scaled_val_in = val_in.copy()
    scaled_val_in[input_cols[1:-1]] = scaled_val_arr
    time_steps = scaled_val_in['Time [s]'].nunique()
    print(time_steps)
    # Run multiple GECKO experiments in parallel
    cluster = LocalCluster(processes=True, n_workers=args.n_workers, threads_per_worker=args.threads_per_worker)
    client = Client(cluster)
    models, predictions, metrics = {}, {}, {}
    for model_type in config["model_configurations"].keys():
        if model_type == 'single_ts_models':
            for model_name in config['model_configurations'][model_type].keys():
                seq_length = 1

                for member in range(ensemble_members):
                    nnet_path = '{}models/{}_{}_{}/'.format(output_path, species, model_name, member)
                    mod = GeckoBoxEmulator(neural_net_path=nnet_path, output_scaler=y_scaler,
                                           input_scaler=x_scaler, input_cols=input_cols, output_cols=output_cols)
                    
                    box_preds = mod.run_ensemble(client=client, data=scaled_val_in, out_data=val_out,
                                                 num_timesteps=time_steps, num_exps=num_exps)
                    y_true, y_preds = match_true_exps(truth=val_out, preds=box_preds, num_timesteps=time_steps,
                                                      seq_length=seq_length)
                    metrics[model_name + '_{}'.format(member)] = ensembled_box_metrics(y_true, y_preds)
                    plot_mae_ts(y_true, y_preds, output_path, model_name, species)
                    predictions[model_name + '_{}'.format(member)] = y_preds
                plot_ensemble(truth=y_true, preds=predictions, output_path=output_path,
                              species=species, model_name=model_name)

        elif model_type == 'multi_ts_models':
            for model_name in config['model_configurations'][model_type].keys():
                seq_length = config['seq_length']
                predictions = {}
                for member in range(ensemble_members):
                    nnet_path = '{}models/{}_{}_{}/'.format(output_path, species, model_name, member)
                    mod = GeckoBoxEmulatorTS(neural_net_path=nnet_path, output_scaler=y_scaler, seq_length=seq_length,
                                             input_cols=input_cols, output_cols=output_cols)
                    box_preds = mod.run_ensemble(client=client, data=scaled_val_in, out_data=val_out,
                                                 num_timesteps=time_steps, num_exps=num_exps)
                    y_true, y_preds = match_true_exps(truth=val_out, preds=box_preds, num_timesteps=time_steps,
                                                      seq_length=seq_length)
                    metrics[model_name + '_{}'.format(member)] = ensembled_box_metrics(y_true, y_preds)
                    predictions[model_name + '_{}'.format(member)] = y_preds
                    plot_mae_ts(y_true, y_preds, output_path, model_name, species)
                    
                plot_ensemble(truth=y_true, preds=predictions, output_path=output_path,
                              species=species, model_name=model_name)
    # write metrics to file
    metrics_str = [f'{key} : {metrics[key]}' for key in metrics]
    with open('{}metrics/{}_box_results.txt'.format(output_path, species), 'a') as f:
        [f.write(f'{st}\n') for st in metrics_str]
        f.write('\n')
    print('Completed in {0:0.1f} minutes.'.format((time.time() - start) / 60))
    return


if __name__ == "__main__":
    main()
