from geckoml.models import DenseNeuralNetwork
from geckoml.data import combine_data, split_data, load_combined_data
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler
from geckoml.metrics import ensembled_base_metrics
import time
import joblib
import argparse
import yaml

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
    dir_path = config['dir_path']
    summary_file = config['summary_file']
    aggregate_bins = config['aggregate_bins']
    bin_prefix = config['bin_prefix']
    min_exp = config['min_exp']
    max_exp = config['max_exp']
    input_vars = config['input_vars']
    output_vars = config['output_vars']
    save_models = config['save_models']
    save_data = config['save_data']
    output_path = config['output_path']
    scaler_type = config['scaler_type']

    if save_data:

        # Load GECKO experiment data, split into ML inputs and outputs and persistence outputs
        input_data, output_data = combine_data(dir_path, summary_file, aggregate_bins, bin_prefix,
                                               input_vars, output_vars, min_exp, max_exp, species)

        # Split into training, validation, testing subsets
        in_train, out_train, in_val, out_val, in_test, out_test = split_data(input_data, output_data)

        # Save combined data to disk
        in_train.to_parquet('{}in_train_{}.parquet'.format(output_path, species))
        out_train.to_parquet('{}out_train_{}.parquet'.format(output_path, species))
        in_val.to_parquet('{}in_val_{}.parquet'.format(output_path, species))
        out_val.to_parquet('{}out_val_{}.parquet'.format(output_path, species))
        in_test.to_parquet('{}in_test_{}.parquet'.format(output_path, species))
        out_test.to_parquet('{}out_test_{}.parquet'.format(output_path, species))

    else:

        in_train, out_train, in_val, out_val, in_test, out_test = load_combined_data(output_path, species)

    # Rescale training and validation / testing data
    x_scaler, y_scaler = scalers[scaler_type](), scalers[scaler_type]()
    scaled_in_train = x_scaler.fit_transform(in_train.iloc[:, 1:-1])
    scaled_in_val = x_scaler.transform(in_val.iloc[:, 1:-1])
    scaled_out_train = y_scaler.fit_transform(out_train.iloc[:, 1:-1])
    scaled_out_val = y_scaler.transform(out_val.iloc[:, 1:-1])

    # Train ML models
    models = {}
    for model_name, model_config in config["model_configurations"].items():

        models[model_name] = DenseNeuralNetwork(**model_config)
        models[model_name].fit(scaled_in_train, scaled_out_train)

    # Calculate validation and testing scores
    metrics = {}
    for model_name in config["model_configurations"].keys():

        preds = models[model_name].predict(scaled_in_val)
        preds = preds.reshape(scaled_out_val.shape[0], scaled_out_val.shape[1])
        transformed_preds = y_scaler.inverse_transform(preds)
        metrics[model_name] = ensembled_base_metrics(out_val, transformed_preds)

    print(metrics)

    # Save ML models and scaling values to disk
    if save_models:

        for model_name in config["model_configurations"].keys():
            models[model_name].save_fortran_model(output_path + model_name + ".nc")
            models[model_name].model.save(output_path + model_name)

        joblib.dump(x_scaler, '{}{}_X.scaler'.format(output_path, species))
        joblib.dump(y_scaler, '{}{}_Y.scaler'.format(output_path, species))

    print('Completed in {0:0.1f} seconds'.format(time.time() - start))
    
    return


if __name__ == "__main__":
    
    main()
