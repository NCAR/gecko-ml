import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import itertools
import functools
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

from earth_mover_dist import EarthMoverDist2D

from multiprocessing import Pool, cpu_count
import multiprocessing


def work(pair):
    this_label, next_label = pair
    with open(f"../toluene/{this_label}.pkl", "rb") as fid:
        x, this_y = pickle.load(fid)
    with open(f"../toluene/{next_label}.pkl", "rb") as fid:
        x, next_y = pickle.load(fid)
    dist = EarthMoverDist2D(this_y, next_y)        
    return this_label, next_label, dist
    

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: python train.py num_workers worker_id")
        sys.exit()
        
    n_workers = int(sys.argv[1])
    worker_id = int(sys.argv[2])
    
    #all_pair_combos = np.array_split(list(itertools.combinations(range(2000), 2)), n_workers)
    #all_pair_combos = all_pair_combos[worker_id]
    
    with open("leftovers.pkl", "rb") as fid:
        all_pair_combos = pickle.load(fid)
    all_pair_combos = np.array_split(all_pair_combos, 5)
    all_pair_combos = all_pair_combos[worker_id]
    
    try:
        with Pool(8) as p:
            for result in tqdm(p.imap(work, all_pair_combos), total = len(all_pair_combos)):
                with open(f"data/{worker_id}.txt", "a+") as fid:
                    this_label, next_label, dist = result
                    fid.write(f"{this_label} {next_label} {dist}\n")
            
    except KeyboardInterrupt:
        sys.exit()