import argparse
import pandas as pd
import yaml
import time
from geckoml.box import GeckoBoxEmulator
from geckoml.metrics import ensembled_box_metrics, mae_time_series, match_true_exps

start = time.time()


def main():
    # read YAML config as provided arg
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="agg_config.yml", help="Path to config file")
    args = parser.parse_args()
    with open(args.config) as config_file:
        config = yaml.load(config_file)

    # Extract config arguments
    species = config['species']
    output_path = config['output_path']
    x_scaler = output_path + config['x_scaler']
    y_scaler = output_path + config['y_scaler']
    nnet = output_path + config['nnet']
    time_steps = config['time_steps']
    num_exps = config['num_exps']
    output_cols = config["output_vars"]

    val_in = pd.read_parquet('{}in_val_{}.parquet'.format(output_path, species))
    val_out = pd.read_parquet('{}out_val_{}.parquet'.format(output_path, species))

    # Run multiple GECKO experiments in parallel
    mod = GeckoBoxEmulator(neural_net_path=nnet, input_scaler_path=x_scaler, output_scaler_path=y_scaler)
    box_preds = mod.run_ensemble(data=val_in, num_timesteps=time_steps, num_exps=num_exps)

    y_true, y_preds = match_true_exps(truth=val_out, preds=box_preds, num_timesteps=time_steps)

    hd, rmse = ensembled_box_metrics(y_true, y_preds)
    print('Hellenger Distance: {}'.format(hd))
    print('RMSE: {}'.format(rmse))

    mae = mae_time_series(y_true, y_preds, output_cols)
    ax = mae.plot()
    ax.set_title('MAE per Timestep')
    fig = ax.get_figure()
    fig.savefig('{}{}mae_timeseries.png'.format(output_path, species), bbox_inches='tight')

    print('Completed in {0:0.1f} seconds'.format(time.time() - start))
    return


if __name__ == "__main__":
    main()