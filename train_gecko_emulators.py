from geckoml.models import DenseNeuralNetwork
from geckoml.data import combine_data, split_data, repopulate_scaler
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import argparse
import yaml

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
    min_exp = config['min_exp']
    max_exp = config['max_exp']
    input_vars = config['input_vars']
    output_vars = config['output_vars']
    save_fortran = config['save_fortran']
    output_path = config['output_path']
    

    # Load GECKO experiment data, split into ML inputs and outputs and persistence outputs
    input_data, output_data = combine_data(dir_path, summary_file, aggregate_bins, bin_prefix,
                                           input_vars, output_vars, min_exp, max_exp, species)
    
    # Split into training, validation, testing subsets
    in_train, out_train, in_val, out_val, in_test, out_test = split_data(input_data, output_data)

    # Rescale training and validation / testing data
    scaled_in_train = repopulate_scaler(in_train).fit_transform(in_train.iloc[:,:-1])
    scaled_in_val = repopulate_scaler(in_val).fit_transform(in_val.iloc[:,:-1])
    scaled_out_train = repopulate_scaler(out_train).fit_transform(out_train.iloc[:,:-1])
    scaled_out_val = repopulate_scaler(out_val).fit_transform(out_val.iloc[:,:-1])
    
    # Train ML models
    dnn = DenseNeuralNetwork(hidden_layers=2, hidden_neurons=100,lr=0.0005,batch_size=512, epochs=2)
    dnn.fit(scaled_in_train,scaled_out_train)

    # Calculate validation and testing scores
    preds = pd.DataFrame(dnn.predict(scaled_in_val).reshape(-1,scaled_out_train.shape[1]))
    
    # Save ML models, scaling values, and verification data to disk
    if save_fortran:
        dnn.save_fortran_model(output_path)
    
    print('Completed in {0:0.1f} seconds'.format(time.time() - start))
    
    return

if __name__ == "__main__":
    
    main()
