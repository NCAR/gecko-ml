import argparse
import pandas as pd
import yaml
import time
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler
import tensorflow as tf
from geckoml.box import GeckoBoxEmulator, GeckoBoxEmulatorTS
from geckoml.metrics import ensembled_box_metrics, plot_mae_ts, match_true_exps
from dask.distributed import Client, LocalCluster

gpus = tf.config.experimental.list_physical_devices('GPU')
for device in gpus:
    tf.config.experimental.set_memory_growth(device, True)

start = time.time()
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
    output_path = config['output_path']
    x_scaler = output_path + config['x_scaler']
    y_scaler = output_path + config['y_scaler']
    num_exps = config['num_exps']
    input_cols = config['input_vars']
    output_cols = config['output_vars']

    # Read validation data and scaler objects
    val_in = pd.read_parquet('{}in_val_{}.parquet'.format(output_path, species))
    val_out = pd.read_parquet('{}out_val_{}.parquet'.format(output_path, species))

    x_scaler = joblib.load(x_scaler)
    y_scaler = joblib.load(y_scaler)

    scaled_val_arr = x_scaler.transform(val_in.iloc[:, 1:-1])
    scaled_val_in = val_in.copy()
    scaled_val_in[input_cols[1:-1]] = scaled_val_arr
    time_steps = val_in['Time [s]'].nunique()

    # Run multiple GECKO experiments in parallel
    cluster = LocalCluster(processes=True, n_workers=72, threads_per_worker=1)
    client = Client(cluster)
    models, metrics = {}, {}
    for model_type in config["model_configurations"].keys():

        if model_type == 'single_ts_models':

            for model_name in config['model_configurations'][model_type].keys():
                seq_length = 1
                nnet_path = '{}{}/'.format(config['output_path'], model_name)
                mod = GeckoBoxEmulator(neural_net_path=nnet_path, output_scaler=y_scaler,
                                       input_scaler=x_scaler)
                box_preds = mod.run_ensemble(client=client, data=scaled_val_in, num_timesteps=time_steps,
                                             num_exps=num_exps)
                y_true, y_preds = match_true_exps(truth=val_out, preds=box_preds, num_timesteps=time_steps,
                                                  seq_length=seq_length)
                metrics[model_name] = ensembled_box_metrics(y_true, y_preds)
                plot_mae_ts(y_true, y_preds, output_path, model_name)

        elif model_type == 'multi_ts_models':

            for model_name in config['model_configurations'][model_type].keys():
                seq_length = config['seq_length']
                nnet_path = '{}{}/'.format(config['output_path'], model_name)
                mod = GeckoBoxEmulatorTS(neural_net_path=nnet_path, output_scaler=y_scaler, seq_length=seq_length,
                                         input_cols=input_cols, output_cols=output_cols)
                box_preds = mod.run_ensemble(client=client, data=scaled_val_in, num_timesteps=time_steps,
                                             num_exps=num_exps)
                y_true, y_preds = match_true_exps(truth=val_out, preds=box_preds, num_timesteps=time_steps,
                                                  seq_length=seq_length)
                metrics[model_name] = ensembled_box_metrics(y_true, y_preds)
                plot_mae_ts(y_true, y_preds, output_path, model_name)

    client.shutdown()

    # write metrics to file
    metrics_str = [f'{key} : {metrics[key]}' for key in metrics]
    with open('{}box_results.txt'.format(output_path), 'a') as f:
        [f.write(f'{st}\n') for st in metrics_str]
        f.write('\n')

    print('Completed in {0:0.1f} seconds'.format(time.time() - start))
    return


if __name__ == "__main__":
    main()
