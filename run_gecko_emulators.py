import numpy as np
import argparse
import joblib
import pandas as pd
import yaml
import time
from geckoml.box import GeckoBoxEmulator
from geckoml.data import get_starting_conds
from dask.distributed import Client, LocalCluster

start = time.time()

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
    input_vars = config['input_vars']
    output_vars = config['output_vars']
    output_path = config['output_path']
    x_scaler = output_path + config['x_scaler']
    y_scaler = output_path + config['y_scaler']
    nnet = output_path + config['nnet']

    exp = 'Exp1989'
    total_timesteps = 1439

    # single experiment example
    # We will want to run multiple GECKO experiments in parallel

    val_in = pd.read_parquet('{}in_val_{}.parquet'.format(output_path, species))
    val_out = pd.read_parquet('{}out_val_{}.parquet'.format(output_path, species))

    starting_conditions = get_starting_conds(data=val_in, exp=exp)
    print(starting_conditions.shape)
    output_scaler = joblib.load(x_scaler)
    scaled_sc = output_scaler.transform(starting_conditions.iloc[:,1:-1])



    mod = GeckoBoxEmulator(neural_net_path=nnet, output_scaler_path=y_scaler)
    box_preds = mod.predict(scaled_sc, total_timesteps)

    print('Completed in {0:0.1f} seconds'.format(time.time() - start))
    print(box_preds.shape)

    return

    # We will want to run multiple GECKO experiments in parallel

    #client = Client()
    #lient.map()


if __name__ == "__main__":
    main()