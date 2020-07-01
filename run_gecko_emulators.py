import numpy as np
import argparse
import yaml
import time
from geckoml.box import GeckoBoxEmulator
from geckoml.data import get_starting_conds

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
    
    exp_num = 299
    start_timestep = 0
    total_timesteps = 1439
    
    # single experiment example 
    # We will want to run multiple GECKO experiments in parallel
    
    starting_conditions = get_starting_conds(dir_path, summary_file, bin_prefix, input_vars,
                    output_vars, aggregate_bins, species, exp_num, start_timestep)

    mod = GeckoBoxEmulator(neural_net_path=nnet, input_scaler_path=x_scaler, output_scaler_path=y_scaler)
    box_preds = mod.predict(starting_conditions, total_timesteps)
    
    print('Completed in {0:0.1f} seconds'.format(time.time() - start))
    print(box_preds.shape)
    
    return

if __name__ == "__main__":
    main()
