import pandas as pd
import numpy as np
import logging
import random
import pickle
import torch
import glob
import json
import yaml
import time
import tqdm
import re
import os

from multiprocessing import Pool

from sklearn.preprocessing import (StandardScaler, RobustScaler, 
                                   MaxAbsScaler, MinMaxScaler, QuantileTransformer)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from os.path import join, exists


logger = logging.getLogger(__name__)


class LoadGeckoData:
    
    def __init__(self, 
                 data_path,
                 save_path,
                 summary_file,  
                 bin_prefix = [], 
                 input_vars = [],
                 output_vars = [],
                 seq_length = 1, 
                 num_timesteps = 1439,
                 experiment_subset = [],
                 cached_dir = "./",
                 shuffle = True,
                 scaler_x = None,
                 scaler_y = None, 
                 fit = False, 
                 *args, **kwargs):
        
        self.path = data_path
        self.save_path = save_path
        self.summary_file = summary_file
        self.experiment_subset = experiment_subset
        self.cached_dir = cached_dir
        self.bin_prefix = bin_prefix
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.seq_length = 1
        self.num_timesteps = num_timesteps
        
        self.shuffle = shuffle
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.fit = fit
        
        self.load()
        self.on_epoch_end()
        
        self.reshape = True
        if any([self.scaler_x is None, self.scaler_y is None]) or self.fit:
            self.fit_scalers()
            
    def load(self):
        file_list = glob.glob(os.path.join(self.path, 'ML2019_*'))
        self.file_list = sorted(file_list, key = lambda x: int(x.split("Exp")[1].strip(".csv")))
        self.summary_file = pd.read_csv(
            os.path.join(self.path, self.summary_file), skiprows = 3
        )
        self.summary_file.columns = [x.strip() for x in self.summary_file.columns]
    
    def get_transform(self):
        return self.scaler_x, self.scaler_y

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.experiment_subset)
    
    def __getitem__(self, idx):
        'Generate one data point'
        
        ### Find the relevant experiment file 
        exp = self.experiment_subset[idx]
        
        ### If we have this experiment already cached, load it
        cached = f"{self.cached_dir}/{exp}.pkl"
        if os.path.isfile(cached) and self.scaler_x is not None:
            with open(cached, "rb") as fid:
                return pickle.load(fid)
        
        ### Else we have to create it for the first time.
        for file_path in self.file_list:
            if f"Exp{exp}" in file_path:
                break

        ### Load the experiment file
        df = pd.read_csv(file_path)
        df.columns = [x.strip() for x in df.columns]

        ### Load the summary file
        exp_num = int(re.findall("_Exp(\d+)*", file_path)[0])
        summary_file = self.summary_file[(
            self.summary_file['id'] == f"Exp{exp_num}"
        )].copy()

        for variable in summary_file.columns:
            df[variable] = summary_file[variable][exp_num]

        if len(self.bin_prefix) > 0: 
            for prefix in self.bin_prefix:
                df[prefix] = df.loc[:, df.columns.str.contains(prefix, regex=False)].sum(axis=1)

        input_subset = df[self.input_vars].iloc[:-1,:].copy()
        output_subset = df[self.output_vars].iloc[1:,:].copy()

        if "index" in input_subset.columns:
            input_subset = input_subset.drop(columns = ["index"])
        if "index" in output_subset.columns:
            output_subset = output_subset.drop(columns = ["index"])

        input_subset = self.add_diurnal_signal(input_subset)

        self.processed += 1
        if self.processed == self.__len__():
            self.on_epoch_end()

        if self.scaler_x is not None:
            input_subset = self.scaler_x.transform(
                input_subset.drop(['Time [s]', 'id'], axis=1)
            )
        if self.scaler_y is not None:
            output_subset = self.scaler_y.transform(
                output_subset.drop(['Time [s]', 'id'], axis=1)
            )
        if self.reshape:
            input_subset, output_subset = self.reshape_data(input_subset, output_subset)
            
            with open(cached, "wb") as fid:
                pickle.dump([input_subset, output_subset], fid)

        return input_subset, output_subset
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.processed = 0
        if self.shuffle == True:
            random.shuffle(self.file_list)
            
    def add_diurnal_signal(self, x_data):
        """
        Apply Function to static temperature to add +- 4 [K] diurnal signal (dependent of time [s] of timeseries).
        Args:
            x_data: Pre-scaled/normalized input data (Pandas DF).

        Returns: Same df with function applied to temperature feature.
        """
        x_data['temperature (K)'] = x_data['temperature (K)'] + 4.0 * np.sin(
            (x_data['Time [s]'] * 7.2722e-5 + (np.pi / 2.0 - 7.2722e-5 * 64800.0)))

        return x_data
    
    def get_tendencies(self, df):
        """
         Transform dataframe to time tendencies rather than actual values. Preserves static environmental values.
        Args:
            df: Pre-scaled input dataframe.
            input_cols: Input columns to be transformed (should include 'id' and 'Time' for indexing).
        Returns: Pandas dataframe with input columns transformed to tendencies (Removes the first sample of each Exp).
        """
        df_copy = df.copy()
        dummy_df = df[self.output_vars].drop(['Time [s]'], axis=1).groupby('id').diff().reset_index(drop=True)
        df_copy[self.output_vars[1:-1]] = dummy_df.values
        df_copy.loc[:, ~df_copy.columns.isin(self.output_vars)] = df.loc[:, ~df.columns.isin(self.output_vars)]
        dff = df_copy.groupby('id').apply(lambda x: x.iloc[1:, :]).reset_index(drop=True)
        return dff
    
    def fit_scalers(self):

        filepath = os.path.join(self.save_path, "scalers.pkl")
        
        if os.path.isfile(filepath) and not self.fit:
            #logging.info(f"Loading data preprocessing models from {filepath}")
            with open(filepath, "rb") as fid:
                self.num_timesteps, self.scaler_x, self.scaler_y = pickle.load(fid)
            
        else:
            logger.info("Fitting data preprocessing models: QuantileTransformer")
            self.reshape = False                    
            with Pool(8) as p:
                xs, ys = zip(*[(x,y) for (x,y) in tqdm.tqdm(
                    p.imap(self.__getitem__, range(len(self.experiment_subset))),
                    total = len(self.experiment_subset))
                ])
            p.join()
            p.close()                    
                    
            xs = pd.concat(xs)
            ys = pd.concat(ys)
            self.reshape = True
            
            self.num_timesteps = xs['Time [s]'].nunique()

            self.scaler_x = Pipeline(
                steps=[('quant', QuantileTransformer()), ('minmax', MinMaxScaler((0, 1)))]
            )
            self.scaler_y = Pipeline(
                steps=[('quant', QuantileTransformer()), ('minmax', MinMaxScaler((0, 1)))]
            )
            
            scaled_in_train = self.scaler_x.fit_transform(
                xs.drop(['Time [s]', 'id'], axis=1)
            )
            scaled_out_train = self.scaler_y.fit_transform(
                ys.drop(['Time [s]', 'id'], axis=1)
            )

            with open(filepath, "wb") as fid:
                pickle.dump([self.num_timesteps, self.scaler_x, self.scaler_y], fid)
            
            logger.info(f"Saved data preprocessing models to {filepath}")
            
    def reshape_data(self, x_data, y_data):
        """
        Reshape matrix data into sample shape for LSTM training.

        Args:
            x_data: DataFrame containing input features (columns) and time steps (rows).
            y_data: Matrix containing output features (columns) and time steps (rows).
            seq_length: Length of look back time steps for one time step of prediction.
            num_timesteps: Number of time_steps per experiment.

        Returns: Two np.ndarrays, the first of shape (samples, length of sequence,
            number of features), containing the input data for the LSTM. The second
            of shape (samples, number of output features) containing the expected output for each input
            sample.
        """

        x_data = torch.from_numpy(x_data.astype(np.float32))
        y_data = torch.from_numpy(y_data.astype(np.float32))
                
        return x_data, y_data    