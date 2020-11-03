import warnings
warnings.filterwarnings("ignore")

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


if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print("Usage: python train.py /path/to/config.yml")
        sys.exit()
    
    ############################################################
    
    with open(sys.argv[1]) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        
    ############################################################
    #
    # Create the save directory if it does not exist
    #
    ############################################################
    
#     try:
#         os.makedirs(config["log"])
#     except:
#         pass
    
#     try:
#         os.makedirs(config["data"]["cached_dir"])
#     except:
#         pass
    
#     shutil.copyfile(sys.argv[1], os.path.join(config["log"], sys.argv[1]))
    
    ############################################################
    #
    # Load a logger
    #
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
    logger_name = os.path.join(config["log"], "log.txt")
    fh = logging.FileHandler(logger_name,
                             mode="w",
                             encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    ############################################################
    #
    # Set the device to a cuda-enabled GPU or the cpu
    #
    ############################################################

    is_cuda = torch.cuda.is_available()
    device = torch.device(torch.cuda.current_device()) if is_cuda else torch.device("cpu")
    
    if is_cuda:
        torch.backends.cudnn.benchmark = True
    
    logging.info(f'Preparing to use device {device}')
    
    ############################################################
    #
    # Create train/test/val splits
    #
    ############################################################
    
    # A fixed random seed for reproducability
    if "seed" in config:
        random.seed(config["seed"])
    else:
        random.seed(5000)
    
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
    
    logging.info(f"Loading training data iterator using {config['iterator']['num_workers']} workers")
    
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
    
    trainer = BaseTrainer(
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
        logging.info(
           f"Loaded ReduceLROnPlateau learning rate annealer with patience {schedule_config['patience']}"
        )
    elif "ExponentialLR" in config["callbacks"]:
        schedule_config = config["callbacks"]["ExponentialLR"]
        scheduler = ExponentialLR(trainer.optimizer, **schedule_config)
        logging.info(
           f"Loaded ExponentialLR learning rate annealer with reduce factor {schedule_config['gamma']}"
        )

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
            
    trainer.train(scheduler, early_stopping, metrics_logger)