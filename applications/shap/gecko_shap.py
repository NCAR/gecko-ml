import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.keras.backend.set_floatx('float64')
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import multiprocessing as mp
import tqdm.auto as tqdm
import pandas as pd
import numpy as np
import logging
import joblib
import yaml
import shap
import time
import yaml
import sys
import os 

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from argparse import ArgumentParser
from os.path import join
from typing import List


def args():
    parser = ArgumentParser(description=
                            "Options for computing SHAP values for GECKO models"
                            )

    parser.add_argument("model_config", type=str, help=
    "Path to the model configuration containing your inputs."
                        )
    parser.add_argument(
        "-s",
        "--save_path",
        dest="save_path",
        type=str,
        default="./",
        help="Where to save the SHAP results"
    )
    parser.add_argument(
        "-n",
        "--n_background",
        dest="n_background",
        type=int,
        default=1000,
        help="Number of background points selected from the training data"
    )
    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=1,
        help="Multiprocessing option: The number of worker subsets to create from experiment list"
    )
    parser.add_argument(
        "--worker",
        dest="worker",
        type=int,
        default=1,
        help="The subset of experiments to be analyzed by this worker (the ID of the worker)"
    )
    parser.add_argument(
        "-m",
        "--model_weights",
        dest="model_weights",
        type=str,
        default="./",
        help="Path to the model weights to be used in SHAP analysis"
    )
    return vars(parser.parse_args())


def box_val_shap(mod: tf.python.keras.engine.training.Model, 
                 exps: List[str], 
                 num_timesteps: int,
                 in_array: np.ndarray, 
                 env_array: np.ndarray):
        
    """
    Evaluate the model in box mode on experiments to obtain an
    array holding the time-ordered predictions for use in SHAP.
    
    Inputs:
    mod: Keras Model object
    - Model to use to predict on the validation experiments
    exps: List[str]
    - List of experiment names
    num_timesteps: int
    - number of timesteps to evaluate
    in_array: np.array
    - Array containing the truth data
    env_array: np.array 
    - Array containing the environmental settings
    
    Returns: np.array
    - Array containing the inputs to the model at each time step
    """
    
    logging.info("Preparing the validation data")

    # use initial condition @ t = 0 and get the first prediction
    pred_array = np.empty((len(exps), 1439, 9))
    pred = mod.predict(in_array[:, 0, :])
    pred_array[:, 0, :] = in_array[:, 0, :]

    # use the first prediction to get the next, and so on for num_timesteps
    for i in tqdm.tqdm(range(1, num_timesteps)):
        temperature = in_array[:, i, 3:4]
        static_env = env_array[:, -5:]
        new_input = np.block([pred, temperature, static_env])
        pred = mod(new_input, training=False)
        pred_array[:, i, :] = new_input

    return pred_array


def results(exp_id: str, 
            shap_results: List[np.ndarray], 
            save_location: str):
    
    """
    Method for creating SHAP plots and saving the results to disk
    
    Inputs: 
    exp_id: string
    - String identity of the experiment (Exp1600)
    shap_results: List[np.array]
    - A list containing the SHAP result at each time step.
    save_location: string
    - Leading filepath to the save directory

    Returns: None
    """
    
    logging.info(f"Post-processing the SHAP results for experiment {exp_id}")

    pre = np.vstack([i.squeeze(0)[:, 0] for i in shap_results])
    gas = np.vstack([i.squeeze(0)[:, 1] for i in shap_results])
    air = np.vstack([i.squeeze(0)[:, 2] for i in shap_results])

    plt.figure(figsize=(7, 14))

    plt.subplot(311)
    leg = ['gas', 'aerosol', 'temperature', 'sza', 'pre-existing aerosol', 'ozone', 'nox', 'OH']
    for i in [1, 2, 3, 4, 5, 6, 7, 8]:
        plt.plot(range(1439), pre[:, i])
    plt.ylabel("Precursor SHAP", fontsize=14)
    plt.legend(leg, fontsize=12, ncol=2, loc='best')

    plt.subplot(312)
    leg = ['precursor', 'aerosol', 'temperature', 'sza', 'pre-existing aero', 'ozone', 'nox', 'OH']
    for i in [0, 2, 3, 4, 5, 6, 7, 8]:
        plt.plot(range(1439), gas[:, i])
    plt.ylabel("Gas SHAP", fontsize=14)
    plt.legend(leg, fontsize=12, ncol=2, loc='best')

    plt.subplot(313)
    leg = ['precursor', 'gas', 'temperature', 'sza', 'pre-existing aero', 'ozone', 'nox', 'OH']
    for i in [0, 1, 3, 4, 5, 6, 7, 8]:
        plt.plot(range(1439), air[:, i])
    plt.ylabel("Aerosol SHAP", fontsize=14)
    plt.legend(leg, fontsize=12, ncol=2, loc='best')

    plt.xlabel("Time (s)", fontsize=14)

    plt.tight_layout()

    plt.savefig(f"{save_location}/validation_{exp_id}.png")
    np.save(f"{save_location}/validation_{exp_id}_pre.npy", pre)
    np.save(f"{save_location}/validation_{exp_id}_gas.npy", gas)
    np.save(f"{save_location}/validation_{exp_id}_aero.npy", air)


class Experiment:
    
    """
    Object to assist with computing SHAP results
    
    Methods:
    self.work computes the SHAP quantities for the prediction output tasks at a time t.
    """

    def __init__(self, 
                 x: np.ndarray, 
                 exp_id: int, 
                 explainer: shap.explainers._exact.Exact):
            
        """
        Inputs:
        x: np.array
        - Array containing the model predictions for all experiments
        exp_id: string
        - String name of an experiment
        explainer: SHAP object
        - Object for computing the SHAP values
        """
        
        self.x = x
        self.id = int(exp_id)
        self.explainer = explainer

    def work(self, t):
        """
        Inputs:
        t: integer
        - The time-step to compute SHAP on the model predictions
        
        Returns: np.array
        - SHAP values for experiment self.id at timestep t.
        """
        
        return self.explainer(np.expand_dims(self.x[self.id][t], axis=0)).values


if __name__ == '__main__':

    args_dict = args()
    config_file = args_dict.pop("model_config")

    if not os.path.isfile(config_file):
        logging.warning(
            f"You must supply a valid configuration. The file at {config_file} does not exist."
        )
        sys.exit(1)

    # Load the model configuration
    with open(config_file) as fid:
        conf = yaml.load(fid, Loader=yaml.FullLoader)

    # Grab the parser options for the SHAP settings
    shap_save_location = args_dict.pop("save_path")
    n_background = args_dict.pop("n_background")
    workers = args_dict.pop("workers")
    worker = args_dict.pop("worker")
    model_weights = args_dict.pop("model_weights")

    # Config check some of the SHAP settings
    if not os.path.isdir(shap_save_location):
        logging.info(f"Creating the directory {shap_save_location} to save SHAP results")
        os.mkdir(shap_save_location)

    if not os.path.isdir(model_weights):
        logging.warning(
            f"Failed to load the model because the directory {model_weights} does not exist. Exiting."
        )
        sys.exit(1)

    ############################################################
    # Set up a logger

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Save the log file
    logger_name = os.path.join(shap_save_location, "log.txt")
    fh = logging.FileHandler(logger_name,
                             mode="w",
                             encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    ############################################################

    # Load conf args
    species = conf['species']
    output_path = conf['output_path']
    exps = conf['box_val_exps']
    input_cols = conf['input_vars']
    output_cols = conf['output_vars']
    columns = ['Precursor [ug/m3]', 'Gas [ug/m3]', 'Aerosol [ug_m3]']

    # Load the data
    try:
        in_train = pd.read_parquet(join(output_path, 'validation_data', f'{species}_in_train.parquet'))
        out_train = pd.read_parquet(join(output_path, 'validation_data', f'{species}_out_train.parquet'))
        in_val = pd.read_parquet(join(output_path, 'validation_data', f'{species}_in_val.parquet'))
    except:
        in_train = pd.read_csv(f'/glade/scratch/cbecker/gecko_data/{species}_train_in_agg.csv')
        out_train = pd.read_csv(f'/glade/scratch/cbecker/gecko_data/{species}_train_out_agg.csv')
        in_val = pd.read_csv(f'/glade/scratch/cbecker/gecko_data/{species}_val_in_agg.csv')

    num_timesteps = in_train['Time [s]'].nunique()

    try:
        x_scaler = joblib.load(join(output_path, 'scalers', f'{species}_x.scaler'))
        y_scaler = joblib.load(join(output_path, 'scalers', f'{species}_y.scaler'))
    except:
        # Rescale training and validation / testing data
        scalers = {"MinMaxScaler": MinMaxScaler, "StandardScaler": StandardScaler}
        scaler_type = conf['scaler_type']
        if scaler_type == "MinMaxScaler":
            x_scaler = scalers[scaler_type]((conf['min_scale_range'], conf['max_scale_range']))
        else:
            x_scaler = scalers[scaler_type]()
            x_scaler = x_scaler.fit(in_train.drop(['Time [s]', 'id'], axis=1))

    scaled_in_train = x_scaler.transform(in_train.drop(['Time [s]', 'id'], axis=1))


    # Prepare the data to be used in "box" mode
    def work(exp):
        in_data = x_scaler.transform(in_val[in_val['id'] == exp].iloc[:, 1:-1])
        env_conds = in_data[0, -6:]
        return np.expand_dims(in_data, axis=0), np.expand_dims(env_conds, axis=0)


    val_exps = list(in_val['id'].unique())
    pool = mp.Pool(processes=8)
    val_in_array, val_env_array = zip(*[result for result in tqdm.tqdm(pool.imap(work, val_exps), total=len(val_exps))])
    pool.close()
    # val_in_array, val_env_array = zip(*[work(exp) for exp in tqdm.tqdm(val_exps)])

    val_in_array = np.concatenate(val_in_array)  # (num_experiments, num_timesteps, outputs)
    val_env_array = np.concatenate(val_env_array)

    # Load the pretrained model
    model = load_model(model_weights, compile=False)

    # Prepare the validation data by running the model in box mode, return the time-ordered predictions
    _x = box_val_shap(
        model,
        val_exps,
        num_timesteps,
        val_in_array,
        val_env_array
    )

    # Take a background selection from the training data
    background_random_selection = np.random.choice(scaled_in_train.shape[0], n_background, replace=False)
    background = shap.maskers.Independent(scaled_in_train[background_random_selection])
    
    # Initialize the SHAP object to compute SHAP values relative to the background
    explainer = shap.Explainer(model, background)

    val_experiments = {int(k): n for k, n in enumerate(list(in_val['id'].unique()))}
    val_experiments = sorted(list(val_experiments.items()))

    # Option to use multiple workers (nodes)
    if workers > 1:
        logging.info(f"Multiprocessing being used with {workers} workers, I am worker {worker + 1} / {workers}")
        val_experiments = np.array_split(val_experiments, workers)
        val_experiments = val_experiments[worker]

    run_times = []
    for k, (idx, exp) in enumerate(val_experiments):
        # Perform SHAP analysis on one experiment
        t0 = time.time()
        logging.info(f"Starting experiment {exp}, {k + 1} / {len(val_experiments)}")
        experiment = Experiment(_x, idx, explainer)
        shap_results = [experiment.work(t) for t in tqdm.tqdm(range(_x.shape[1]))]
        results(exp, shap_results, shap_save_location)

        # Log the time it took to perform SHAP analysis on 1 experiment
        runtime = time.time() - t0
        run_times.append(runtime)
        logging.info(f"Completed experiment {exp}. This took {runtime} s")

        # Estimate how much time remains
        remaining = len(val_experiments) - len(run_times)
        average_runtime = np.mean(run_times)
        time_to_finish = (average_runtime * remaining) / 60.0
        logging.info(f"There are {remaining} experiments remaining, estimated time left: {time_to_finish} mins")