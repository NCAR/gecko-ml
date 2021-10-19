import warnings

warnings.filterwarnings("ignore")

import os
import sys
import yaml
import copy
import torch
import joblib
import logging
import subprocess
import pandas as pd

from collections import defaultdict
from argparse import ArgumentParser

from geckoml.models import GRUNet
from geckoml.data import load_data, transform_data, save_scaler_csv
from geckoml.box import rnn_box_train_one_epoch, rnn_box_test

# Get the GPU
is_cuda = torch.cuda.is_available()
device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")
if is_cuda:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

# Set the default logger
logger = logging.getLogger(__name__)


def prepare_pbs_launch_script(model_config: str, worker: int, workers_per_node: int):
    """ Create a list of commands to send to the PBS scheduler from the model configuration
    Args:
        model_config (str)
    Returns:
        pbs_options (List[str])
    """

    pbs_options = ["#!/bin/bash -l"]
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
    if workers_per_node > 1:
        for copy in range(workers_per_node):
            model_id = workers_per_node * worker + copy
            save_loc = os.path.join(model_config["output_path"], f"models/model_{model_id}.yml")
            pbs_options.append(f"python {gecko_path} -c {save_loc} &")
        pbs_options.append("wait")
    else:
        save_loc = os.path.join(model_config["output_path"], f"models/model_{worker}.yml")
        pbs_options.append(f"python {gecko_path} -c {save_loc}")
    return pbs_options


def submit_workers(model_conf, workers, workers_per_node=1):
    """ Submit launch scripts to the PBS scheduler
    Args:
        model_conf (Dict[str, str])
        workers (int)
    """

    total = 0
    for worker in range(workers):

        for subworker in range(workers_per_node):
            # Grab the parent save location for the models
            conf = copy.deepcopy(model_conf)
            script_path = os.path.join(conf["output_path"], f"models")
            # conf["output_path"] = script_path
            conf["pbs"]["batch"]["o"] = os.path.join(script_path, f"out_{worker}")
            conf["pbs"]["batch"]["e"] = os.path.join(script_path, f"err_{worker}")
            conf["model_configurations"]["RNN"]["GRU_1"]["member"] = total

            # Save the updated conf to the new directory
            with open(f'{script_path}/model_{total}.yml', 'w') as outfile:
                logger.info(f"Saving an ensemble configuration to {script_path}/model_{total}.yml")
                yaml.dump(conf, outfile, default_flow_style=False)

            total += 1

        # Prepare the launch script pointing to the new config file.
        logger.info(f"Preparing the launch script for worker {worker}")
        launch_script = prepare_pbs_launch_script(conf, worker, workers_per_node)

        # Save the configured script
        script_location = os.path.join(script_path, f"launch_pbs_{worker}.sh")
        logger.info(f"Saving worker {worker} launch script to {script_location}")
        with open(script_location, "w") as fid:
            for line in launch_script:
                fid.write(f"{line}\n")

        # Launch the pbs/slurm job
        name_condition = "N" in conf["pbs"]["batch"]
        slurm_job_name = conf["pbs"]["batch"]["N"] if name_condition else "gecko_gru"

        w = subprocess.Popen(
            f"qsub -N {slurm_job_name}_{worker} {script_location}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
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
    member = conf["model_configurations"]["RNN"]["GRU_1"]["member"]

    # Validation starting times
    start_times = rnn_conf["validation_starting_times"]

    # Load the data
    logger.info(f"Loading the train and validation data for {species}, this may take a few minutes")

    for folder in ['models', 'plots', 'metrics']:
        os.makedirs(os.join(output_path, folder), exist_ok=True)

    data = load_data(data_path, aggregate_bins, species, input_vars, output_vars, log_trans_cols)

    transformed_data, x_scaler, y_scaler = transform_data(
        data,
        output_path,
        species,
        tendency_cols,
        log_trans_cols,
        scaler_type,
        output_vars,
        train=True
    )

    joblib.dump(x_scaler, os.join(output_path, 'models', f'{species}_x.scaler'))
    joblib.dump(y_scaler, os.join(output_path, 'models', f'{species}_y.scaler'))
    save_scaler_csv(x_scaler, input_vars, output_path, species, scaler_type)

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

    # Load the model 
    logger.info("Loading a 1-step GRU model")
    model = GRUNet(hidden_dim, n_layers, rnn_dropout)
    model.build(input_size, output_size)
    model = model.to(device)

    # Load the train and test losses
    logger.info("Loading the train and validation loss criterion (Huber and MAE respectively)")
    criterion = torch.nn.SmoothL1Loss()  # Huber loss
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
        patience=lr_patience,
        verbose=True,
        min_lr=1.0e-13
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
            train_in_array,
            out_col_idx,
            hidden_weight=hidden_weight,
            loss_weights=loss_weights
        )

        step_val_loss, val_loss, val_pearson, _, _ = rnn_box_test(
            model,
            val_criterion,
            val_in_array,
            data['val_out'],
            y_scaler,
            output_vars,
            val_out_col_idx,
            log_trans_cols,
            tendency_cols,
            stable_thresh=10,
            start_times=start_times
        )

        # Get the last learning rate
        learning_rate = optimizer.param_groups[0]['lr']

        logger.info(
            f"Epoch: {epoch} train_loss: {train_loss:.6f} val_loss: {val_loss:.6f} val_step_loss: {step_val_loss:.6f} lr: {learning_rate}")

        # Put things into a results dictionary
        results_dict["epoch"].append(epoch)
        results_dict["train_loss"].append(train_loss)
        results_dict["val_loss"].append(val_loss)
        results_dict["val_step_loss"].append(step_val_loss)
        results_dict["lr"].append(learning_rate)
        df = pd.DataFrame.from_dict(results_dict).reset_index()

        # Save the dataframe to disk
        df.to_csv(os.path.join(conf["output_path"], f"models/training_log_{member}.csv"), index=False)

        # update schedulers
        metric = "val_step_loss"

        # Update the scheduler and anneal the learning rate if required
        lr_scheduler.step(results_dict[metric][-1])

        # Save the model if its the best so far.
        if results_dict[metric][-1] == min(results_dict[metric]):
            state_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss
            }
            torch.save(state_dict, os.path.join(conf["output_path"], f"models/{species}_gru_{member}.pt"))

        # Stop training if we have not improved after X epochs
        best_epoch = [i for i, j in enumerate(results_dict[metric]) if j == min(results_dict[metric])][0]
        offset = epoch - best_epoch
        if offset >= stopping_patience:
            break

    # Select the best loss
    best_box_metric = min(results_dict[metric])

    # Return box_mae to optuna
    results = {
        "box_metric": best_box_metric
    }

    logger.info(f"Completed training, best validation metric was {best_box_metric}")

    return results


if __name__ == '__main__':

    parser = ArgumentParser(
        description="Train N 1-step GRU models using w nodes and t threads per node, where N = w * t"
    )
    parser.add_argument(
        "-c",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs."
    )
    parser.add_argument(
        "-w",
        dest="workers",
        type=int,
        default=1,
        help="The number of GPU nodes (workers) to use to train model(s). Default is 1."
    )
    parser.add_argument(
        "-t",
        dest="threads",
        type=int,
        default=1,
        help="The number of CPU threads (threads) to use to train model(s). Default is 1."
    )

    args_dict = vars(parser.parse_args())
    config_file = args_dict.pop("model_config")
    workers = int(args_dict.pop("workers"))
    threads = int(args_dict.pop("threads"))
    mode = "B" if workers == 1 else "A"

    if not os.path.isfile(config_file):
        logger.warning(f"The model config does not exist at {config_file}. Failing with error.")
        sys.exit(1)

    with open(config_file) as cf:
        config = yaml.load(cf, Loader=yaml.FullLoader)

    # Create the save directories when submitting 
    for folder in ['models', 'plots', 'metrics']:
        os.makedirs(os.join(config["output_path"], folder), exist_ok=True)

    ############################################################
    # Initialize logger to stream to stdout
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Save the log file
    member = config["model_configurations"]["RNN"]["GRU_1"]["member"]
    logger_name = os.path.join(config["output_path"], f"models/log_{member}.txt")
    fh = logging.FileHandler(logger_name,
                             mode="w",
                             encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    ############################################################
    ### MODE A: Launch w jobs to train w * t models
    if mode == "A":
        submit_workers(config, workers, workers_per_node=threads)

    ### MODE B: Train a model
    if mode == "B":
        results = train(config)
