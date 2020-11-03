import warnings
warnings.filterwarnings("ignore")

import shutil
import sherpa
from sherpa.algorithms import Genetic
import sherpa.algorithms.bayesian_optimization as BayesianOptimization

import pandas as pd
import numpy as np
import logging
import random
import pickle
import shutil
import glob
import json
import yaml
import time
import sys
import re
import os
import gc

from holodecml.vae.checkpointer import *
from holodecml.vae.optimizers import *
from holodecml.vae.tqdm import tqdm

from geckoml.torch.data_reader import LoadGeckoData
from geckoml.torch.trainer import BaseTrainer
from geckoml.torch.model import DenseNet

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import *
from torch import nn

import torch.nn.functional as F
import torch

from typing import List, Dict, Callable, Union, Any, TypeVar, Tuple


class CustomTrainer(BaseTrainer):
    
    def train(self,
              study,
              trial,
              scheduler,
              early_stopping,
              metrics_logger):        
        
        flag = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        stop_early = False
        
        for epoch in range(self.start_epoch, self.epochs):
            
            try:
                train_loss = self.train_one_epoch(epoch)
                test_loss = self.test(epoch)
            
            except: # how I am dealing with memory errors ... 
                stop_early = True
                test_loss = 1e9
            
            study.add_observation(
                trial=trial,
                iteration=epoch,
                objective=-test_loss
            )
            
            if study.should_trial_stop(trial) or stop_early:
                break

        return study, trial
    

if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print("Usage: python train.py /path/to/config.yml")
        sys.exit()
    
    ############################################################
    
    with open(sys.argv[1]) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        
    ############################################################
        
        
    is_cuda = torch.cuda.is_available()
    device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")
    if is_cuda:
        torch.backends.cudnn.benchmark = True

    print(f'Preparing to use device {device}')

    ############################################################

    parameters = [
        #sherpa.Discrete('batch_size', [1, 128]),
        sherpa.Discrete('n_hidden', [1, 5]),
        sherpa.Discrete('hidden_size1', [1, 20000]),
        sherpa.Discrete('hidden_size2', [1, 1000]),
        #sherpa.Discrete('window_size', [1, 1440]),
        sherpa.Continuous('lr', [1e-3, 1e-7], scale='log')
    ]

    #algorithm = BayesianOptimization(max_num_trials=100) #Genetic(max_num_trials=100)
    algorithm = BayesianOptimization.GPyOpt(
        max_concurrent=1,
        model_type='GP_MCMC',
        acquisition_type='EI_MCMC',
        max_num_trials=100
    )

    study = sherpa.Study(
        parameters=parameters,
        algorithm=algorithm,
        output_dir=config["log"],
        lower_is_better=False
    )

    #######################################################################

    for trial in study:

        print("Trial {}:\t{}".format(trial.id, trial.parameters))

        # Update the configuration
        config["model"]["hidden_dims"] = [int(trial.parameters['hidden_size1'])]
        if trial.parameters['n_hidden'] > 1:
            for k in range(int(trial.parameters['n_hidden'])-1):
                config["model"]["hidden_dims"].append(int(trial.parameters['hidden_size2']))
        config["model"]["dropouts"] = [
            0.0 for k in range(len(config["model"]["hidden_dims"]))
        ]
        config["trainer"]["window_size"] = 1440 # int(trial.parameters['window_size'])
        config["optimizer"]["lr"] = float(trial.parameters['lr'])
        config["trainer"]["epochs"] = 2
        config["iterator"]["batch_size"] = 1 # int(trial.parameters["batch_size"])
        bs = config["iterator"]["batch_size"]

        ############################################################
        #
        # Create train/test/val splits
        #
        ############################################################

#         # A fixed random seed for reproducability
#         if "seed" in config:
#             random.seed(config["seed"])
#         else:
#             random.seed(5000)

        file_list = glob.glob(os.path.join(config["data"]["data_path"], 'ML2019_*'))
        file_list = sorted(file_list, key = lambda x: int(x.split("Exp")[1].strip(".csv")))

        experiments = []
        for x in file_list:
            x = int(x.split("Exp")[1].strip(".csv"))
            if x >= config["data"]["min_exp"] and x <= config["data"]["max_exp"]:
                experiments.append(x)

        train_split, _test = train_test_split(experiments, test_size = 0.2)
        valid_split, test_split = train_test_split(_test, test_size = 0.5)

        # Save the cached data for fast look-up
        cached_data = glob.glob(config["data"]["cached_dir"] + "/*pkl")

        ############################################################
        #
        # Load the train/test/val splits
        #
        ############################################################

        # Load the data sets
        train_data_set = LoadGeckoData(
            **config["data"],
            experiment_subset = train_split,
            shuffle = True,
            fit = False if len(cached_data) else True
        )

        valid_data_set = LoadGeckoData(
            **config["data"],
            num_timesteps = train_data_set.num_timesteps,
            experiment_subset = valid_split,
            shuffle = False,
            scaler_x = train_data_set.scaler_x,
            scaler_y = train_data_set.scaler_y 
        )

        test_data_set = LoadGeckoData(
            **config["data"],
            num_timesteps = train_data_set.num_timesteps,
            experiment_subset = test_split,
            shuffle = False,
            scaler_x = train_data_set.scaler_x,
            scaler_y = train_data_set.scaler_y 
        )

        ############################################################
        #
        # Load the data iterators 
        #
        ############################################################

        train_dataloader = DataLoader(
            train_data_set,
            **config["iterator"]
        )

        config["iterator"]["batch_size"] = len(valid_split)

        valid_dataloader = DataLoader(
            valid_data_set,
            **config["iterator"]
        )

        test_dataloader = DataLoader(
            test_data_set,
            **config["iterator"]
        )

        ############################################################
        #
        # Load the model class
        #
        ############################################################

        model = DenseNet(**config["model"])

        if is_cuda:
            model = model.to(device)

        ############################################################
        #
        # Load the optimizer class
        #
        ############################################################

        optimizer = LookaheadDiffGrad(
            model.parameters(),
            lr=config["optimizer"]["lr"],
            weight_decay=1e-5
        )

        ############################################################
        #
        # Load the trainer class
        #
        ############################################################

        trainer = CustomTrainer(
            model, 
            optimizer,
            train_data_set, 
            valid_data_set, 
            train_dataloader, 
            valid_dataloader,
            device = device,
            **config["trainer"]
        )

        ############################################################
        #
        # Load callbacks
        #
        ############################################################

        # Initialize LR annealing scheduler 
        if "ReduceLROnPlateau" in config["callbacks"]:
            schedule_config = config["callbacks"]["ReduceLROnPlateau"]
            scheduler = ReduceLROnPlateau(trainer.optimizer, **schedule_config)
        elif "ExponentialLR" in config["callbacks"]:
            schedule_config = config["callbacks"]["ExponentialLR"]
            scheduler = ExponentialLR(trainer.optimizer, **schedule_config)

        # Early stopping
        checkpoint_config = config["callbacks"]["EarlyStopping"]
        early_stopping = EarlyStopping(**checkpoint_config)

        # Write metrics to csv each epoch
        metrics_logger = MetricsLogger(**config["callbacks"]["MetricsLogger"])

        ############################################################
        #
        # Train a model
        #
        ############################################################

        study, trial = trainer.train(study, trial, scheduler, early_stopping, metrics_logger)
        study.finalize(trial=trial)                

        # Save the parameters and the trial.id 
        save_path = config["log"]
        with open(os.path.join(save_path, "stats.txt"), "a+") as fid:
            fid.write(f"{trial.id} {bs} {trial.parameters}\n")
            fid.write(f"Best results: {study.get_best_result()}\n")
       
        study.results.to_csv(os.path.join(save_path, "results.csv"))