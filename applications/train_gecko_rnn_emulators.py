import warnings
warnings.filterwarnings("ignore")

import os
import sys
import yaml
import copy
import torch
import pickle
import joblib
import logging
import traceback
import subprocess
import numpy as np
import pandas as pd
#import tqdm.auto as tqdm
import multiprocessing as mp

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import defaultdict
from argparse import ArgumentParser
    
from geckoml.models import GRUNet
from geckoml.metrics import *
from geckoml.data import *
from geckoml.box import *


# Get the GPU
is_cuda = torch.cuda.is_available()
device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")
if is_cuda:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    
# Set the default logger
logger = logging.getLogger(__name__)



def prepare_pbs_launch_script(model_config: str, worker: int):
    
    """ Create a list of commands to send to the PBS scheduler from the model configuration
    Args:
        model_config (str)
    Returns:
        pbs_options (List[str])
    """
    
    pbs_options = ["#!/bin/bash -l"]
    save_loc = os.path.join(model_config["output_path"], f"models/model_{worker}.yml")
    for arg, val in model_config["pbs"]["batch"].items():
        if arg == "l" and type(val) == list:
            for opt in val:
                pbs_options.append(f"#PBS -{arg} {opt}")
        elif len(arg) == 1:
            pbs_options.append(f"#PBS -{arg} {val}")
        else:
            pbs_options.append(f"#PBS --{arg}={val}")     
    if "bash" in model_config["pbs"]:
        if len(model_config["pbs"]["bash"]) > 0:
            for line in model_config["pbs"]["bash"]:
                pbs_options.append(line)
    if "kernel" in model_config["pbs"]:
        if model_config["pbs"]["kernel"] is not None:
            pbs_options.append(f'{model_config["pbs"]["kernel"]}')
    gecko_path = os.path.realpath(__file__)
    pbs_options.append(f"python {gecko_path} -c {save_loc}")
    return pbs_options


def submit_workers(model_conf, workers):
    
    """ Submit launch scripts to the PBS scheduler
    Args:
        model_conf (Dict[str, str])
        workers (int)
    """
    
    for worker in range(workers):
        # Grab the parent save location for the models
        conf = copy.deepcopy(model_conf)
        script_path = os.path.join(conf["output_path"], f"models")
        #conf["output_path"] = script_path
        conf["pbs"]["batch"]["o"] = os.path.join(script_path, f"out_{worker}")
        conf["pbs"]["batch"]["e"] = os.path.join(script_path, f"err_{worker}")
        conf["model_configurations"]["single_ts_models"]["gru"]["member"] = worker
        
#         # Check if directories exist
#         if os.path.isdir(script_path):
#             # If yes, tell the user they need to delete the directory and try again
#             logger.warning(
#                 f"You must remove cached data at {script_path} before this script can run. Exiting."
#             )
#             sys.exit(1)
#         # Else no, make the new directory. 
#         os.mkdir(script_path)
#         logger.info(f"Creating a new directory at {script_path}")
        
        # Save the updated conf to the new directory
        with open(f'{script_path}/model_{worker}.yml', 'w') as outfile:
            logger.info(f"Saving a modified configuration (model.yml) to {script_path}")
            yaml.dump(conf, outfile, default_flow_style=False)
        
        # Prepare the launch script pointing to the new config file.
        logger.info(f"Preparing the launch script for worker {worker}")
        launch_script = prepare_pbs_launch_script(conf, worker)
        
        # Save the configured script
        logger.info(f"Saving the launch script (launch_pbs.sh) to {script_path}")
        script_location = os.path.join(script_path, f"launch_pbs_{worker}.sh")
        with open(script_location, "w") as fid:
            for line in launch_script:
                fid.write(f"{line}\n")

        # Launch the slurm job
        name_condition = "N" in conf["pbs"]["batch"]
        slurm_job_name = conf["pbs"]["batch"]["N"] if name_condition else "gecko_rnn_train"
        
        w = subprocess.Popen(
            f"qsub -N {slurm_job_name}_{worker} {script_location}",
            shell=True,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE
        ).communicate()
        job_id = w[0].decode("utf-8").strip("\n")
        logger.info(
            f"Submitted pbs batch job {worker + 1}/{workers} with id {job_id}"
        )
        
        # Write the job ids to file for reference
        with open(os.path.join(script_path, "job_id.txt"), "w") as fid:
            fid.write(f"{job_id}\n")



def train(conf):
    
    """ Train a 1-step GRU model using the parameters from a configuration file
    Args:
        conf (Dict[str, str])
    Returns:
        results_dict (Dict[str, float])
    """
    
    logger.info("Reading the model configuration")

    scalers = {
        "MinMaxScaler": MinMaxScaler,
        "StandardScaler": StandardScaler
    }
    
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
    #ensemble_members = conf["ensemble_members"]
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
    
    lr_patience = rnn_conf["lr_patience"]
    stopping_patience = rnn_conf["stopping_patience"]
    member = conf["model_configurations"]["single_ts_models"]["gru"]["member"]

    ### Want to be able to reliably test L2 penalty = 0, but the 
    ### optuna loguniform prevents this -- so truncate below a certain value
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
    
#     in_train = pd.read_csv(f'/glade/scratch/cbecker/gecko_data/{species}_train_in_agg.csv')
#     out_train = pd.read_csv(f'/glade/scratch/cbecker/gecko_data/{species}_train_out_agg.csv')
#     in_val = pd.read_csv(f'/glade/scratch/cbecker/gecko_data/{species}_val_in_agg.csv')
#     out_val = pd.read_csv(f'/glade/scratch/cbecker/gecko_data/{species}_val_out_agg.csv')

#     in_train = in_train.drop(columns = [x for x in in_train.columns if x == "Unnamed: 0"])
#     out_train = out_train.drop(columns = [x for x in out_train.columns if x == "Unnamed: 0"])
#     in_val = in_val.drop(columns = [x for x in in_val.columns if x == "Unnamed: 0"])
#     out_val = out_val.drop(columns = [x for x in out_val.columns if x == "Unnamed: 0"])

    num_timesteps = in_train['Time [s]'].nunique()

    # Rescale training and validation / testing data
    if scaler_type == "MinMaxScaler":
        x_scaler = scalers[scaler_type]((conf['min_scale_range'], conf['max_scale_range']))
    else:
        x_scaler = scalers[scaler_type]()

    logger.info("Transforming the data using a standard scaler")
    scaled_in_train = x_scaler.fit_transform(in_train.drop(['Time [s]', 'id'], axis=1))
    scaled_in_val = x_scaler.transform(in_val.drop(['Time [s]', 'id'], axis=1))

    y_scaler = get_output_scaler(x_scaler, output_vars, scaler_type, data_range=(
        conf['min_scale_range'], conf['max_scale_range']))

    scaled_out_train = y_scaler.transform(out_train.drop(['Time [s]', 'id'], axis=1))
    scaled_out_val = y_scaler.transform(out_val.drop(['Time [s]', 'id'], axis=1))
    
    joblib.dump(x_scaler, os.path.join(output_path, f'scalers/{species}_{member}_x.scaler'))
    joblib.dump(y_scaler, os.path.join(output_path, f'scalers/{species}_{member}_y.scaler'))

    y = partition_y_output(scaled_out_train, 1, aggregate_bins)
    y_val = partition_y_output(scaled_out_val, 1, aggregate_bins)
    
    # Validation starting times
    start_times = [0, 10, 50, 100, 500, 1000]
    
    # Batch the training experiments 
    logger.info("Batching the training data by experiment, this may take a few minutes")
    def work(exp):
        in_data = x_scaler.transform(in_train[in_train['id'] == exp].iloc[:, 1:-1])
        env_conds = in_data[0, -6:]
        return (np.expand_dims(in_data, axis=0), np.expand_dims(env_conds, axis=0))
    train_exps = list(in_train['id'].unique())
    in_array, env_array = zip(*[work(x) for x in tqdm(train_exps)])
    in_array = np.concatenate(in_array) # (num_experiments, num_timesteps, outputs)
    env_array = np.concatenate(env_array)

    logger.info("Batching the validation data by experiment")
    def work(exp):
        in_data = x_scaler.transform(in_val[in_val['id'] == exp].iloc[:, 1:-1])
        env_conds = in_data[0, -6:]
        return (np.expand_dims(in_data, axis=0), np.expand_dims(env_conds, axis=0))
    val_exps = list(in_val['id'].unique())
    val_in_array, val_env_array = zip(*[work(x) for x in tqdm(val_exps)])
    val_in_array = np.concatenate(val_in_array) # (num_experiments, num_timesteps, outputs)
    val_env_array = np.concatenate(val_env_array)
    
    # Get the shapes of the input and output data 
    input_size = in_array.shape[-1]
    output_size = in_array.shape[-1] - env_array.shape[-1]

    # Load the model 
    logger.info("Loading a 1-step GRU model")
    model = GRUNet(hidden_dim, n_layers, rnn_dropout)
    model.build(input_size, output_size)
    model = model.to(device)

    # Load the train and test losses
    logger.info("Loading the train and validation loss criterion (Huber and MAE respectively)")
    criterion = torch.nn.SmoothL1Loss() # Huber loss
    val_criterion = torch.nn.L1Loss()  # Mean absolute error

    # Load an optimizer
    logger.info("Loading the Adam optimizer")
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Load an scheduler for the RNN model
    logger.info(f"Annealing the learning rate using on-plateau with epoch-patience {lr_patience}")
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience = lr_patience, 
        verbose = True,
        min_lr = 1.0e-13
    )

    # Train the model 
    logger.info("Training and validating the model")

    results_dict = defaultdict(list)
    
    for epoch in range(epochs):
    
        train_loss, model, optimizer = rnn_box_train_one_epoch(
            model, 
            optimizer, 
            criterion, 
            batch_size, 
            train_exps, 
            num_timesteps, 
            in_array, 
            env_array,
            hidden_weight = hidden_weight
        )

        val_loss, step_val_loss, _, _ = rnn_box_test(
            model, 
            val_exps, 
            num_timesteps, 
            val_in_array, 
            val_env_array, 
            y_scaler, 
            output_vars, 
            out_val,
            start_times = start_times
        )

        # Get the last learning rate
        learning_rate = optimizer.param_groups[0]['lr']

        # Put things into a results dictionary
        results_dict["epoch"].append(epoch)
        results_dict["train_loss"].append(train_loss)
        results_dict["val_loss"].append(val_loss)
        results_dict["step_val_loss"].append(step_val_loss)
        results_dict["lr"].append(learning_rate)
        df = pd.DataFrame.from_dict(results_dict).reset_index()

        # Save the dataframe to disk
        df.to_csv(os.path.join(conf["output_path"], f"models/training_log_{member}.csv"), index = False)

        logger.info(
            f"Epoch: {epoch} train_loss: {train_loss:.6f} val_loss: {val_loss:.6f} step_val_loss: {step_val_loss:.6f} lr: {learning_rate}"
        )

        # Update the scheduler and anneal the learning rate if required
        lr_scheduler.step(val_loss)
        
        # Save the model if its the best so far.
        if val_loss == min(results_dict["val_loss"]):
            state_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss
            }
            torch.save(state_dict, os.path.join(conf["output_path"], f"models/{species}_gru_{member}.pt"))
                        
        # Stop training if we have not improved after X epochs
        best_epoch = [i for i,j in enumerate(results_dict["val_loss"]) if j == min(results_dict["val_loss"])][0]
        offset = epoch - best_epoch
        if offset >= stopping_patience:
            break

    # Select the best loss
    best_box_mae = min(results_dict["val_loss"])

    # Return box_mae to optuna
    results = {
        "box_mae": best_box_mae
    }

    logger.info(f"Completed training, best loss was {best_box_mae}")

    return results


if __name__ == '__main__':
    
    parser = ArgumentParser(
        description="Train N 1-step GRU models using N nodes, where N is integer"
    )
    parser.add_argument(
        "-c",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs."
    )
    parser.add_argument(
        "-n",
        dest="workers",
        type=int,
        default=1,
        help="The number of nodes (workers) to use to train model(s). Default is 1."
    )
    
    args_dict = vars(parser.parse_args())
    config_file = args_dict.pop("model_config")
    workers = int(args_dict.pop("workers"))
    mode = "B" if workers == 1 else "A"
    
    if not os.path.isfile(config_file):
        logger.warning(f"The model config does not exist at {config_file}. Failing with error.")
        sys.exit(1)
        
    with open(config_file) as cf:
        config = yaml.load(cf, Loader=yaml.FullLoader)
        
        
#     # Check if directories exist
#     if os.path.isdir(script_path):
#         # If yes, tell the user they need to delete the directory and try again
#         logger.warning(
#             f"You must remove cached data at {script_path} before this script can run. Exiting."
#         )
#         sys.exit(1)
#     # Else no, make the new directory. 
#     os.mkdir(script_path)
#     logger.info(f"Creating a new directory at {script_path}")
        
    # Create the save directories when submitting 
    for folder in ['models', 'plots', 'validation_data', 'metrics', 'scalers']:
        os.makedirs(join(config["output_path"], folder), exist_ok=True)

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
    member = config["model_configurations"]["single_ts_models"]["gru"]["member"]
    logger_name = os.path.join(config["output_path"], f"models/log_{member}.txt")
    fh = logging.FileHandler(logger_name,
                             mode="w",
                             encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    ############################################################
    ### MODE A: Launch N jobs to train N models
    if mode == "A":
        submit_workers(config, workers)
    
    ### MODE B: Train a model
    if mode == "B":
        results = train(config)