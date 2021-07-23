import warnings
warnings.filterwarnings("ignore")

import os
import sys
import yaml
import glob
import torch
import joblib
import random
import logging
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

from scipy.signal import tukey
from numpy.fft import fft, fftshift
from numpy.fft import rfft, rfftfreq

from geckoml.data import *
from geckoml.models import GRUNet
from geckoml.box import rnn_box_test
from geckoml.metrics import (ensembled_metrics, match_true_exps, plot_ensemble, 
                             plot_bootstrap_ci, plot_crps_bootstrap, plot_unstability, 
                             plot_scatter_analysis, fourier_analysis)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from functools import partial
import tqdm


# Get the GPU
is_cuda = torch.cuda.is_available()
device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")


# Set the default logger
logger = logging.getLogger(__name__)

# How many CPUs available for multiprocessing
n_cpus = min(os.cpu_count(), 8)


def seed_everything(seed=1234):
    """
    Set seeds for determinism
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True


def worker(replica, 
           species = None, 
           output_path = None, 
           hidden_dim = None, 
           n_layers = None, 
           val_exps = None, 
           num_timesteps = None, 
           val_in_array = None, 
           val_env_array = None, 
           y_scaler = None, 
           output_vars = None, 
           out_val = None):
    
    """
    Load GRU ensemble model, run box simulations, and compute performance metrics
    Args:
        replica: ensemble member ID
        species: Modeled species
        output_path: Output path (str)
        hidden_dim: size of the GRU hidden state
        n_laers: number of hidden layers in the GRU
        val_exps: List of experiment names
        num_timesteps: Total time-steps in experiments
        val_in_array: validation input data to the GRU 
        val_env_array: validation environmental data used as input to the GRU
        y_scaler: sklearn scaler transformation
        output_vars: List of names of the prediction outputs
        out_val: output validation data 
    """
    
    logger.info(f"Loading ensemble member {replica} model weights")
     
    # Get the shapes of the input and output data 
    input_size = val_in_array.shape[-1]
    output_size = val_in_array.shape[-1] - val_env_array.shape[-1]

    # Load the model 
    model = GRUNet(hidden_dim, n_layers, 0.0)
    model.build(input_size, 
                output_size,
                os.path.join(output_path, f"models/{species}_gru_{replica}.pt"))
    
    # Predict on the validation split
    logger.info(f"Running box simulations for all experiments using model {replica}")
    box_mae, scaled_box_mae, y_preds, y_true = rnn_box_test(
                    model, 
                    val_exps, 
                    num_timesteps, 
                    val_in_array, 
                    val_env_array, 
                    y_scaler, 
                    output_vars, 
                    out_val
                )
    
    # compute metrics
    performance_results = ensembled_metrics(y_true, y_preds, replica)
    
    # add extra field to the pred and truth arrays to be used later
    y_true['member'] = replica
    y_preds['member'] = replica
    
    # put the results into a dictionary
    results_dict = {
        "replica": replica,
        "y_preds": y_preds,
        "y_true": y_true,
        "metrics": performance_results
    }
    
    # return the results
    return results_dict



if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print("Usage: python run_gecko_rnn_emulators.py model.yml")
        sys.exit()
    
    config_file = str(sys.argv[1])
    
    if not os.path.isfile(config_file):
        logger.warning(f"The model config does not exist at {config_file}. Failing with error.")
        sys.exit(1)
        
    with open(config_file) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    ############################################################
    
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    
    # Save the log file
    logger_name = os.path.join(conf["output_path"], f"metrics/log.txt")
    fh = logging.FileHandler(logger_name,
                             mode="w",
                             encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    ############################################################
    
    seed_everything()
    
    species = conf['species']
    dir_path = conf['dir_path']
    summary_file = conf['summary_file']
    aggregate_bins = conf['aggregate_bins']
    bin_prefix = conf['bin_prefix']
    input_vars = conf['input_vars']
    output_vars = conf['output_vars']
    output_path = conf['output_path']
    scaler_type = conf['scaler_type']
    seq_length = conf['seq_length']
    seed = conf['random_seed']

    # Model settings
    rnn_conf = conf["model_configurations"]["single_ts_models"]["gru"]

    epochs = rnn_conf["epochs"]
    batch_size = rnn_conf["batch_size"]
    learning_rate = rnn_conf["lr"]
    weight_decay = rnn_conf["l2_weight"]
    n_layers = rnn_conf["n_layers"]
    hidden_dim = rnn_conf["hidden_size"]
    rnn_dropout = rnn_conf["rnn_dropout"]
    hidden_weight = rnn_conf["hidden_weight"]
    verbose = rnn_conf["verbose"]
    model_name = "gru"

    lr_patience = rnn_conf["lr_patience"]
    stopping_patience = rnn_conf["stopping_patience"]
    member = conf["model_configurations"]["single_ts_models"]["gru"]["member"]
    weight_decay = weight_decay if weight_decay > 1e-12 else 0.0

    # Load the data
    logger.info(f"Loading the train and validation data for {species}, this may take a few minutes")

    # Load GECKO experiment data, split into ML inputs and outputs and persistence outputs
    input_data, output_data = combine_data(
        dir_path, 
        summary_file, 
        aggregate_bins, 
        bin_prefix, 
        input_vars, 
        output_vars, 
        species
    )

    # Split into training, validation, testing subsets
    in_train, out_train, in_val, out_val, in_test, out_test = split_data(
        input_data=input_data, 
        output_data=output_data, 
        train_start=conf['train_start_exp'],
        train_end=conf['train_end_exp'],
        val_start=conf['val_start_exp'],
        val_end=conf['val_end_exp'],
        test_start=conf['test_start_exp'],
        test_end=conf['test_end_exp']
    )
    
    ####################
    
#     in_train = pd.read_csv(f'/glade/scratch/cbecker/gecko_data/{species}_train_in_agg.csv')
#     out_train = pd.read_csv(f'/glade/scratch/cbecker/gecko_data/{species}_train_out_agg.csv')
#     in_val = pd.read_csv(f'/glade/scratch/cbecker/gecko_data/{species}_val_in_agg.csv')
#     out_val = pd.read_csv(f'/glade/scratch/cbecker/gecko_data/{species}_val_out_agg.csv')

#     in_train = in_train.drop(columns = [x for x in in_train.columns if x == "Unnamed: 0"])
#     out_train = out_train.drop(columns = [x for x in out_train.columns if x == "Unnamed: 0"])
#     in_val = in_val.drop(columns = [x for x in in_val.columns if x == "Unnamed: 0"])
#     out_val = out_val.drop(columns = [x for x in out_val.columns if x == "Unnamed: 0"])
    
    ####################

    num_timesteps = in_train['Time [s]'].nunique()
    num_ensemble_members = len(glob.glob(os.path.join(output_path, 'models/*pt')))
    ensemble_range_ids = list(range(num_ensemble_members))

    # Rescale training and validation / testing data
    x_scaler_path = sorted(glob.glob(os.path.join(output_path, 'scalers/*x.scaler')))[0]
    y_scaler_path = x_scaler_path.replace("x.scaler", "y.scaler")
    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)

    logger.info("Transforming the data using a standard scaler")
    scaled_in_train = x_scaler.fit_transform(in_train.drop(['Time [s]', 'id'], axis=1))
    scaled_in_val = x_scaler.transform(in_val.drop(['Time [s]', 'id'], axis=1))

    scaled_out_train = y_scaler.transform(out_train.drop(['Time [s]', 'id'], axis=1))
    scaled_out_val = y_scaler.transform(out_val.drop(['Time [s]', 'id'], axis=1))

    y = partition_y_output(scaled_out_train, 1, aggregate_bins)
    y_val = partition_y_output(scaled_out_val, 1, aggregate_bins)
    
    # Batch the training experiments 
    logger.info("Batching the training data by experiment, this may take a few minutes")
    def work(exp):
        in_data = x_scaler.transform(in_train[in_train['id'] == exp].iloc[:, 1:-1])
        env_conds = in_data[0, -6:]
        return (np.expand_dims(in_data, axis=0), np.expand_dims(env_conds, axis=0))
    train_exps = list(in_train['id'].unique())

    if n_cpus > 1:
        with mp.Pool(processes=n_cpus) as p:
            in_array, env_array = zip(*[result for result in tqdm.tqdm(p.imap(work, train_exps), total=len(train_exps))])
    else:
        in_array, env_array = zip(*[work(x) for x in tqdm.tqdm(train_exps)])
    in_array = np.concatenate(in_array) # (num_experiments, num_timesteps, outputs)
    env_array = np.concatenate(env_array)

    logger.info("Batching the validation data by experiment")
    def work(exp):
        in_data = x_scaler.transform(in_val[in_val['id'] == exp].iloc[:, 1:-1])
        env_conds = in_data[0, -6:]
        return (np.expand_dims(in_data, axis=0), np.expand_dims(env_conds, axis=0))
    val_exps = list(in_val['id'].unique())
    if n_cpus > 1:
        with mp.Pool(processes=n_cpus) as p:
            val_in_array, val_env_array = zip(*[result for result in tqdm.tqdm(p.imap(work, val_exps), total=len(val_exps))])
    else:
        val_in_array, val_env_array = zip(*[work(x) for x in tqdm.tqdm(val_exps)])
    val_in_array = np.concatenate(val_in_array) # (num_experiments, num_timesteps, outputs)
    val_env_array = np.concatenate(val_env_array)
    
    logger.info(f"Using {n_cpus} workers to run box simulations for {num_ensemble_members} GRU ensemble members")
    
    truth = {}
    predictions = {}
    metrics_dict = {}

    with mp.Pool(n_cpus) as p:
        
        work = partial(worker, 
                       species=species, 
                       output_path=output_path, 
                       hidden_dim=hidden_dim, 
                       n_layers=n_layers, 
                       val_exps=val_exps,
                       num_timesteps=num_timesteps, 
                       val_in_array=val_in_array, 
                       val_env_array=val_env_array, 
                       y_scaler=y_scaler, 
                       output_vars=output_vars, 
                       out_val=out_val)
        
        for results_dict in tqdm.tqdm(p.imap(work, ensemble_range_ids), total = num_ensemble_members):
            replica = results_dict['replica']
            truth[f'gru_{replica}'] = results_dict['y_true']
            predictions[f'gru_{replica}'] = results_dict['y_preds']
            metrics_dict[f'member_{replica}'] = results_dict['metrics']
    
    logger.info("Plotting the ensembled results for random experiments")
    plot_ensemble(
        truth=truth[f"gru_{replica}"], 
        preds=predictions, 
        output_path=output_path, 
        species=species, 
        model_name=model_name
    )
            
    logger.info(f'Saving the predictions to {os.path.join(output_path, "metrics")}')
    all_truth = pd.concat(truth.values())
    all_preds = pd.concat(predictions.values())
    all_preds.to_csv(
        os.path.join(output_path, f'metrics/{species}_{model_name}_preds.csv'), 
        index=False
    )
    all_truth.to_csv(
        os.path.join(output_path, f'metrics/{species}_{model_name}_truth.csv'), 
        index=False
    )
    
    logger.info(f'Saving the performance metrics')
    save_metrics(metrics_dict, output_path, model_name, len(metrics_dict), 'box')
    
    logger.info(f'Creating and saving the bootstrap plot')
    plot_bootstrap_ci(all_truth, all_preds, 
                      output_vars[1:-1], output_path, species, model_name)
    
    logger.info(f'Creating and saving the crps bootstrap plot')
    plot_crps_bootstrap(all_truth, all_preds, 
                        output_vars[1:-1], output_path, species, model_name)
    
    logger.info(f'Creating and saving the instabilities results')
    plot_unstability(all_preds, output_vars[1:-1], output_path, model_name)
    
    logger.info(f'Creating and saving the scatter analysis plot of the training data')
    plot_scatter_analysis(all_preds, all_truth, in_train, in_val, 
                          output_vars[2:-1], output_path, species, model_name)
    
    logger.info(f'Creating and saving the fourier analysis plot using the averaged values for quantites')
    fourier_analysis(all_preds, output_path, species, model_name)
    
    
    
    