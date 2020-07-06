import argparse
import pandas as pd
import yaml
import time
from geckoml.box import GeckoBoxEmulator

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
    output_path = config['output_path']
    x_scaler = output_path + config['x_scaler']
    y_scaler = output_path + config['y_scaler']
    nnet = output_path + config['nnet']

    # We will want to run multiple GECKO experiments in parallel

    val_in = pd.read_parquet('{}in_val_{}.parquet'.format(output_path, species))
    val_out = pd.read_parquet('{}out_val_{}.parquet'.format(output_path, species))

    mod = GeckoBoxEmulator(neural_net_path=nnet, input_scaler_path=x_scaler, output_scaler_path=y_scaler)
    d = mod.run_ensemble(val_in, 100, 7)
    print(d.columns)
    d.to_csv(output_path+'test_output.csv', index=False)
    print('Completed in {0:0.1f} seconds'.format(time.time() - start))

    return

    # We will want to run multiple GECKO experiments in parallel

    #client = Client()
    #lient.map()


if __name__ == "__main__":
    main()