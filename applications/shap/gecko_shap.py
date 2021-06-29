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
import torch
import yaml
import shap
import time
import yaml
import sys
import os 

from captum.attr import GradientShap
from geckoml.models import GRUNet

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from argparse import ArgumentParser
from os.path import join
from typing import List


# Set up the default logger 
logger = logging.getLogger(__name__)


# Set up the torch device globlly
is_cuda = torch.cuda.is_available()
device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")
if is_cuda:
    torch.backends.cudnn.enabled = False
    
    
# Set up the random seed globally
import random
random.seed(5000)


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
    parser.add_argument(
        "-c",
        "--cores",
        dest="cores",
        type=int,
        default=8,
        help="Number of CPU cores available to process the data"
    )
    return vars(parser.parse_args())


def box_val_mlp(model: tf.python.keras.engine.training.Model,
                exps: List[str],
                num_timesteps: int,
                in_array: np.ndarray,
                env_array: np.ndarray) -> (np.array, None):
        
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
    
    logging.info("Preparing the validation data to compute SHAP")
    
    in_array_size = in_array.shape[-1]
    out_array_size = in_array_size - env_array.shape[-1]
    input_array = np.empty((len(exps), num_timesteps, in_array_size))

    # use initial condition @ t = 0 and get the first prediction
    input_array[:, 0, :] = in_array[:, 0, :]
    pred = model(in_array[:, 0, :], training=False)

    # use the first prediction to get the next, and so on for num_timesteps
    for i in tqdm.tqdm(range(1, num_timesteps)):
        temperature = in_array[:, i, 3:4]
        static_env = env_array[:, -5:]
        new_input = np.block([pred, temperature, static_env])
        input_array[:, i, :] = new_input
        pred = model(new_input, training=False)

    return input_array, None # empty hidden state


def box_val_rnn(model: torch.nn.Module,
                exps: List[str],
                num_timesteps: int,
                in_array: np.ndarray,
                env_array: np.ndarray) -> (torch.Tensor, torch.Tensor):
        
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
    
    name = "training" if len(exps) > 200 else "validation"
    logging.info(f"Preparing the {name} data to compute SHAP")
    
    in_array_size = in_array.shape[-1]
    out_array_size = in_array_size - env_array.shape[-1]
    
    # use initial condition @ t = 0 and get the first prediction
    pred_array = np.empty((len(exps), num_timesteps, out_array_size))
    input_array = np.empty((len(exps), num_timesteps, in_array_size))
    hidden_array = np.empty((len(exps), num_timesteps, model.hidden_dim))
    h0 = model.init_hidden(torch.from_numpy(in_array[:, 0, :]).float())
    
    input_array[:, 0, :] = in_array[:, 0, :]
    hidden_array[:, 0, :] = h0.detach().cpu().numpy()
    
    pred, h0 = model.predict(in_array[:, 0, :], h0, return_hidden = True)
    pred_array[:, 0, :] = pred

    # use the first prediction to get the next, and so on for num_timesteps
    for i in tqdm.tqdm(range(1, num_timesteps)):
        temperature = in_array[:, i, 3:4]
        static_env = env_array[:, -5:]
        new_input = np.block([pred, temperature, static_env])
        input_array[:, i, :] = new_input
        hidden_array[:, i, :] = h0.detach().cpu().numpy()
        pred, h0 = model.predict(new_input, h0, return_hidden = True)
        pred_array[:, i, :] = pred
    
    return input_array, hidden_array



def results(exp_id: str, 
            shap_results: List[np.ndarray], 
            save_location: str) -> None:
    
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
    
    pre = np.vstack([i[0] for i in shap_results])
    gas = np.vstack([i[1] for i in shap_results])
    air = np.vstack([i[2] for i in shap_results])

    plt.figure(figsize=(7, 14))
    colors = {i: f"C{i}" for i in range(9)}

    plt.subplot(311)
    leg = ['gas', 'aerosol', 'temperature', 'sza', 'pre-existing aerosol', 'ozone', 'nox', 'OH']
    for i in [1, 2, 3, 4, 5, 6, 7, 8]:
        plt.plot(range(1439), pre[:, i], c = colors[i])
    plt.ylabel("Precursor SHAP", fontsize=14)
    plt.legend(leg, fontsize=12, ncol=2, loc='best')

    plt.subplot(312)
    leg = ['precursor', 'aerosol', 'temperature', 'sza', 'pre-existing aero', 'ozone', 'nox', 'OH']
    for i in [0, 2, 3, 4, 5, 6, 7, 8]:
        plt.plot(range(1439), gas[:, i], c = colors[i])
    plt.ylabel("Gas SHAP", fontsize=14)
    plt.legend(leg, fontsize=12, ncol=2, loc='best')

    plt.subplot(313)
    leg = ['precursor', 'gas', 'temperature', 'sza', 'pre-existing aero', 'ozone', 'nox', 'OH']
    for i in [0, 1, 3, 4, 5, 6, 7, 8]:
        plt.plot(range(1439), air[:, i], c = colors[i])
    plt.ylabel("Aerosol SHAP", fontsize=14)
    plt.legend(leg, fontsize=12, ncol=2, loc='best')

    plt.xlabel("Time (s)", fontsize=14)

    plt.tight_layout()

    plt.savefig(f"{save_location}/validation_{exp_id}.png")
    np.save(f"{save_location}/validation_{exp_id}_pre.npy", pre)
    np.save(f"{save_location}/validation_{exp_id}_gas.npy", gas)
    np.save(f"{save_location}/validation_{exp_id}_aero.npy", air)

    
class CustomNet(GRUNet):
    
    """
    A custom GRUNet from geckoml.models for use in computing SHAP
    
    """
      
    def forward(self, 
                x: torch.Tensor, 
                h: torch.Tensor, 
                return_hidden: bool = False) -> (torch.Tensor, torch.Tensor):
        
        """
            Pass the inputs through the model and return the prediction
        
            Inputs
            x: torch.Tensor
            - The input containing precursor, gas, aerosol, and envionmental values
            h: torch.Tensor
            - The hidden state to the GRU at time t
            
            Returns
            out: torch.Tensor
            - The encoded input
            h [optional]: torch.Tensor
            - The hidden state returned by the GRU at time t + 1
        """
        
        x = x.unsqueeze(1)
        try:
            out, h = self.gru(x, h)
        except:
            h = h.unsqueeze(0)
            out, h = self.gru(x, h)
            
        out = self.fc(self.relu(out[:,-1]))
        
        if return_hidden:
            return out, h
        
        return out
    
    def predict(self, 
                x: np.array, 
                h: np.array, 
                return_hidden: bool = False) -> (np.array, np.array):
        
        """
            Predict method for running the model in box mode.
            Handles converting numpy tensor input to torch
            and moving the data to the GPU
        
            Inputs
            x: np.array
            - The input containing precursor, gas, aerosol, and envionmental values
            h: np.array
            - The hidden state to the GRU at time t
            
            Returns: str
            x: np.array
            - The encoded input
            h: np.array
            - The hidden state to the GRU at time t + 1
        
        """
    
        device = self._device() if self.device is None else self.device
        with torch.no_grad():
            x = torch.from_numpy(x).float().to(device)
            x, h = self.forward(x, h, return_hidden)
        return x.cpu().detach().numpy(), h
    

class Experiment:
    
    """
    Object to assist with computing SHAP results
    
    Methods:
    self.work computes the SHAP quantities for the prediction output tasks at a time t.
    """

    def __init__(self, 
                 x: np.ndarray, 
                 exp_id: int, 
                 explainer: shap.explainers._exact.Exact, 
                 h: bool = None, 
                 background: np.array = None, 
                 n_samples: int = 1000):
            
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
        self.h = h         
        self.id = int(exp_id)
        self.explainer = explainer
        self.n_samples = n_samples 
        self.mode = 'rnn' if self.h is not None else 'mlp'
        
        # for use with captum's explainers
        # where the background is fed into a method rather than 
        # at the initiation as is done with the shap package.
        self.background = tuple(background) 

    def work(self, 
             t: int) -> List[np.array]:
        
        """
        Inputs:
        t: integer
        - The time-step to compute SHAP on the model predictions
        
        Returns: np.array
        - SHAP values for experiment self.id at timestep t.
        """
        
        if self.mode == 'mlp':
            return self.explainer.shap_values(
                np.expand_dims(self.x[self.id][t], axis=0), 
                nsamples = self.n_samples
            )
        
        if self.mode == 'rnn':
            inputs = [
                torch.from_numpy(self.x[self.id][t:(t+1)]).float().to(device),
                torch.from_numpy(self.h[self.id][t:(t+1)]).float().to(device)
            ]
            
            shap_results = []
            for i in range(3):
                (x_att, h_att), approximation_error = self.explainer.attribute(
                    tuple(inputs),
                    target = i,
                    baselines = self.background,
                    n_samples = self.n_samples,
                    return_convergence_delta = True
                )
                shap_results.append(
                    x_att.detach().cpu().numpy()
                )
            return shap_results
            
            # Tested both; captum is faster (see the Jupyter notebook).
            # Both algos converge when nsamples ~=1000, so its going to be slow
            # return [i[0] for i in self.explainer.shap_values(inputs)] #shap method
            

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
    cpu_cores = args_dict.pop("cores")

    # Config check some of the SHAP settings
    if not os.path.isdir(shap_save_location):
        logging.info(f"Creating the directory {shap_save_location} to save SHAP results")
        os.mkdir(shap_save_location)

    if not os.path.isdir(model_weights) and not os.path.isfile(model_weights):
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
        out_val = pd.read_csv(f'/glade/scratch/cbecker/gecko_data/{species}_val_out_agg.csv')
        in_train = in_train.drop(columns = [x for x in in_train.columns if x == "Unnamed: 0"])
        out_train = out_train.drop(columns = [x for x in out_train.columns if x == "Unnamed: 0"])
        in_val = in_val.drop(columns = [x for x in in_val.columns if x == "Unnamed: 0"])
        out_val = out_val.drop(columns = [x for x in out_val.columns if x == "Unnamed: 0"])

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
    
    # Batch the training experiments 
    logger.info("Batching the training data by experiment, this may take a few minutes")
    logger.info("The shape of the output will be: (num_experiments, num_timesteps, n_outputs)")
    def work(exp):
        in_data = x_scaler.transform(in_train[in_train['id'] == exp].iloc[:, 1:-1])
        env_conds = in_data[0, -6:]
        return (np.expand_dims(in_data, axis=0), np.expand_dims(env_conds, axis=0))
    train_exps = list(in_train['id'].unique())
    if cpu_cores > 1:
        with mp.Pool(processes=cpu_cores) as p:
            in_array, env_array = zip(*[
                result for result in tqdm.tqdm(p.imap(work, train_exps), total=len(train_exps))
            ])
    else:
        in_array, env_array = zip(*[work(x) for x in tqdm.tqdm(train_exps)])
    in_array = np.concatenate(in_array) # (num_experiments, num_timesteps, outputs)
    env_array = np.concatenate(env_array)

    logger.info("Batching the validation data by experiment")
    logger.info("The shape of the output will be: (num_experiments, num_timesteps, n_outputs)")
    def work(exp):
        in_data = x_scaler.transform(in_val[in_val['id'] == exp].iloc[:, 1:-1])
        env_conds = in_data[0, -6:]
        return (np.expand_dims(in_data, axis=0), np.expand_dims(env_conds, axis=0))
    val_exps = list(in_val['id'].unique())
    val_in_array, val_env_array = zip(*[work(x) for x in tqdm.tqdm(val_exps)])
    val_in_array = np.concatenate(val_in_array) # (num_experiments, num_timesteps, outputs)
    val_env_array = np.concatenate(val_env_array)

    # Select the model (mlp or rnn) based on the weights file
    rnn_model = True if ".pt" in model_weights else False

    if not rnn_model:
        
        # Load the pretrained MLP model
        model = load_model(model_weights, compile=False)

        # Prepare the validation data by running the model in box mode, return the time-ordered predictions
        _x, _h = box_val_mlp(
            model,
            val_exps,
            num_timesteps,
            val_in_array,
            val_env_array
        )

        # Take a background selection from the training data
        background_random_selection = np.random.choice(scaled_in_train.shape[0], n_background, replace=False)
        background = scaled_in_train[background_random_selection]
        
        # Initialize the SHAP object to compute SHAP values relative to the background
        # https://shap-lrjball.readthedocs.io/en/latest/generated/shap.GradientExplainer.html
        explainer = shap.GradientExplainer(model, background) 
        
    else:
        
        # Load the pretrained GRU model
        input_size = val_in_array.shape[-1]
        output_size = val_in_array.shape[-1] - val_env_array.shape[-1]

        n_layers = conf["rnn_network"]["n_layers"]
        hidden_dim = conf["rnn_network"]["hidden_size"]
        rnn_dropout = conf["rnn_network"]["rnn_dropout"]

        # Using a custom GRU net to handle return of hidden states
        model = CustomNet(hidden_dim, n_layers, rnn_dropout)
        model.build(input_size, output_size)

        # Load the weights from file
        checkpoint = torch.load(
            model_weights,
            map_location=lambda storage, loc: storage
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        # Move the model to device (cpu or gpu)
        model = model.to(device)
        
        # Obtain the inputs to the model for the training split
        # This is used to select the background sample
        t_x, t_h = box_val_rnn(
            model, 
            train_exps, 
            num_timesteps, 
            in_array, 
            env_array
        )
        
        # Obtain inputs to the model for the validation split
        _x, _h = box_val_rnn(
            model, 
            val_exps, 
            num_timesteps,
            val_in_array, 
            val_env_array
        )
        
        # Randomly select background samples from the training data 
        sel1 = np.random.choice(t_x.shape[0], n_background, replace=False)
        sel2 = np.random.choice(t_x.shape[1], n_background, replace=False)

        background_input = t_x[sel1, sel2, :]
        background_hidden = t_h[sel1, sel2, :]
        
        # Set up the multi-input to the GRU as a list [x, h] as required by shap
        background = [
            torch.from_numpy(background_input).float().to(device),
            torch.from_numpy(background_hidden).float().to(device)
        ]
        
        # Initialize the SHAP object to compute SHAP values relative to the background
        # https://captum.ai/api/gradient_shap.html
        explainer = GradientShap(model)
   

    # Set up a sorted list for mapping integers to experiment IDs
    val_experiments = {int(k): n for k, n in enumerate(list(in_val['id'].unique()))}
    val_experiments = sorted(list(val_experiments.items()))

    # Option to use multiple workers (nodes)
    if workers > 1:
        logging.info(
            f"Using {workers} workers (nodes), I am worker {worker + 1} / {workers}"
        )
        val_experiments = np.array_split(val_experiments, workers)
        val_experiments = val_experiments[worker]

    run_times = []
    for k, (idx, exp) in enumerate(val_experiments):
        
        # Perform SHAP analysis on one experiment
        t0 = time.time()
        logging.info(f"Starting experiment {exp}, {k + 1} / {len(val_experiments)}")
        experiment = Experiment(_x, idx, explainer, _h, background)
        shap_results = [experiment.work(t) for t in tqdm.tqdm(range(_x.shape[1]))]

        # Save the results to disk and create some plots
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