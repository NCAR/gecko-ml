import warnings
warnings.filterwarnings("ignore")

import os
import sys
import yaml
import glob
#import torch
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
from geckoml.metrics import (ensembled_metrics, save_analysis_plots, sum_bins)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from argparse import ArgumentParser
from functools import partial
import tqdm


# Get the GPU
#is_cuda = torch.cuda.is_available()
#device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")


# Set the default logger
logger = logging.getLogger(__name__)


# def seed_everything(seed=1234):
#     """
#     Set seeds for determinism
#     """
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.backends.cudnn.benchmark = True
#         torch.backends.cudnn.deterministic = True


def worker(replica, 
           species = None, 
           output_path = None, 
           hidden_dim = None, 
           n_layers = None, 
           val_in_array = None, 
           y_scaler = None, 
           output_vars = None, 
           val_out = None, 
           val_out_col_idx = None,
           log_trans_cols = None,
           tendency_cols = None,
           stable_thresh = 10, 
           start_times = [0]):
    
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

    from geckoml.models import GRUNet
    from geckoml.box import rnn_box_test

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    is_cuda = torch.cuda.is_available()
    device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")

    logger.info(f"Loading ensemble member {replica} model weights")
     
    # Get the shapes of the input and output data 
    input_size = val_in_array.shape[-1]
    output_size = val_out.shape[-1]

    # Load the model 
    model = GRUNet(hidden_dim, n_layers, 0.0)
    model.build(input_size, 
                output_size,
                os.path.join(output_path, f"models/{species}_gru_{replica}.pt"))
    model = model.to(device)
    
    # MAE loss
    val_criterion = torch.nn.L1Loss()
    
    # Predict on the validation split and get the performance metrics
    logger.info(f"Running box simulations for all experiments using model {replica}")
    
    scaled_box_mae, box_mae, metrics, y_preds, y_true = rnn_box_test(
            model, 
            val_criterion,
            val_in_array, 
            val_out,
            y_scaler,
            output_vars, 
            val_out_col_idx,
            log_trans_cols,
            tendency_cols,
            stable_thresh = 10, 
            start_times = start_times
    )
    
    # add extra field to the pred and truth arrays to be used later
    y_true['member'] = replica
    y_preds['member'] = replica
    metrics["ensemble_member"] = replica
    
    # put the results into a dictionary
    results_dict = {
        "replica": replica,
        "y_preds": y_preds,
        "y_true": y_true,
        "metrics": metrics
    }
    
    # return the results
    return results_dict



if __name__ == '__main__':
            
    parser = ArgumentParser(
        description="Run t 1-step GRU models using t threads, where t is integer"
    )
    parser.add_argument(
        "-c",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs."
    )
    parser.add_argument(
        "-t",
        dest="threads",
        type=int,
        default=1,
        help="The number of threads to use to run box simulations. Default is 1."
    )
    
    args_dict = vars(parser.parse_args())
    config_file = args_dict.pop("model_config")
    workers = int(args_dict.pop("workers"))
    
    if not os.path.isfile(config_file):
        logger.warning(f"The model config does not exist at {config_file}. Failing with error.")
        sys.exit(1)
        
    with open(config_file) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
        
    # How many CPUs available for multiprocessing
    n_cpus = min(os.cpu_count(), workers)

    ############################################################
    
    root = logging.getLogger()
    root.setLevel("main")
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
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    ############################################################
    
    #seed_everything()
    
    species = conf['species']
    data_path = conf['dir_path']
    aggregate_bins = conf['aggregate_bins']
    input_vars = conf['input_vars']
    output_vars = conf['output_vars']
    tendency_cols = conf['tendency_cols']
    log_trans_cols = conf['log_trans_cols']
    output_path = conf['output_path']
    scaler_type = conf['scaler_type']
    ensemble_members = conf["ensemble_members"]
    seed = conf['random_seed']
    
    # Get the shapes of the input and output data 
    input_size = len(input_vars)
    output_size = len(output_vars)

    # Model settings
    rnn_conf = conf["model_configurations"]["RNN"]["GRU_1"]
    
    epochs = rnn_conf["epochs"]
    batch_size = rnn_conf["batch_size"]
    learning_rate = rnn_conf["lr"]
    weight_decay = rnn_conf["l2_weight"] if rnn_conf["l2_weight"] > 1e-12 else 0.0
    n_layers = rnn_conf["n_layers"]
    hidden_dim = rnn_conf["hidden_size"]
    rnn_dropout = rnn_conf["rnn_dropout"]
    hidden_weight = rnn_conf["hidden_weight"]    
    loss_weights = [
            rnn_conf["precursor_weight"], 
            rnn_conf["gas_weight"], 
            rnn_conf["aero_weight"]
        ]
    verbose = rnn_conf["verbose"]
    lr_patience = rnn_conf["lr_patience"]
    stopping_patience = rnn_conf["stopping_patience"]
    member = rnn_conf["member"]
    model_name = "GRU"
    
    # Validation starting times
    start_times = rnn_conf["validation_starting_times"]

    # Load the data
    logger.info(f"Loading the train and validation data for {species}, this may take a few minutes")
    
    for folder in ['models', 'plots', 'metrics']:
        os.makedirs(join(output_path, folder), exist_ok=True)

    data = load_data(data_path, aggregate_bins, species, input_vars, output_vars, log_trans_cols)
    
    transformed_data, x_scaler, y_scaler = transform_data(
        data, 
        output_path, 
        species, 
        tendency_cols, 
        log_trans_cols,
        scaler_type, 
        output_vars, 
        train=False
    )
    
    # Batch the training data by experiment
    train_in_array = transformed_data['train_in'].copy()
    n_exps = len(train_in_array.index.unique(level='id'))
    n_timesteps = len(train_in_array.index.unique(level='Time [s]'))
    n_features = len(input_vars)
    out_col_idx = train_in_array.columns.get_indexer(output_vars)
    train_in_array = train_in_array.values.reshape(n_exps, n_timesteps, n_features)
    
    # Batch the validation data by experiment
    val_in_array = transformed_data['val_in'].copy()
    n_exps = len(val_in_array.index.unique(level='id'))
    n_timesteps = len(val_in_array.index.unique(level='Time [s]'))
    val_out_col_idx = val_in_array.columns.get_indexer(output_vars)
    val_in_array = val_in_array.values.reshape(n_exps, n_timesteps, n_features)
    
    ### STOPPED HERE
    n_cpus = min(ensemble_members, n_cpus)
    logger.info(f"Using {n_cpus} workers to run box simulations for {ensemble_members} GRU ensemble members")
    
    truth = {}
    predictions = {}
    metrics = {}

    with mp.Pool(n_cpus) as p:
        work = partial(worker, 
                       species=species, 
                       output_path=output_path, 
                       hidden_dim=hidden_dim, 
                       n_layers=n_layers, 
                       val_in_array=val_in_array, 
                       y_scaler=y_scaler, 
                       output_vars=output_vars, 
                       val_out=data['val_out'],
                       val_out_col_idx=val_out_col_idx,
                       log_trans_cols=log_trans_cols,
                       tendency_cols=tendency_cols,
                       stable_thresh=10,
                       start_times=[0])
        
        for results_dict in tqdm.tqdm(p.imap(work, range(ensemble_members)), total = ensemble_members):
            replica = results_dict['replica']
            truth[f'gru_{replica}'] = results_dict['y_true']
            predictions[f'gru_{replica}'] = results_dict['y_preds']
            metrics[f'member_{replica}'] = results_dict['metrics']
    
    logger.info(f'Saving the predictions to {os.path.join(output_path, "metrics")}')
    all_truth = pd.concat(truth.values())
    all_preds = pd.concat(predictions.values())
    all_preds.to_parquet(join(output_path, f'metrics/{species}_{model_name}_preds.parquet'))
    all_truth.to_parquet(join(output_path, f'metrics/{species}_{model_name}_truth.parquet'))
    
    logger.info("Saving and plotting ensembled performance for random experiments")
    save_metrics(metrics, output_path, model_name, ensemble_members, 'box')
    save_analysis_plots(all_truth, all_preds, data["train_in"], data["val_in"], output_path,
                        output_vars, species, model_name)
    
#     logger.info(f'Creating and saving the fourier analysis plot using the averaged values for quantites')
#     fourier_analysis(all_preds, output_path, species, model_name)