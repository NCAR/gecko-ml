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


from sklearn.preprocessing import (StandardScaler, RobustScaler, 
                                   MaxAbsScaler, MinMaxScaler, QuantileTransformer)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from os.path import join, exists

from multiprocessing import Pool
from functools import partial

logger = logging.getLogger(__name__)


class LoadGeckoData:
    
    def __init__(self, 
                 data_path,
                 save_path,
                 summary_file,  
                 bin_prefix = (), 
                 input_vars = (),
                 output_vars = (),
                 num_timesteps = 1439,
                 experiment_subset = (),
                 min_exp = None,
                 max_exp = None,
                 cached_dir = None,
                 memory_buffer = None,
                 shuffle = True,
                 scaler_x = None,
                 scaler_y = None,
                 log_precursor = False):
        
        self.path = data_path
        self.save_path = save_path
        self.summary_file = summary_file
        self.experiment_subset = experiment_subset

        self.min_exp = min_exp
        self.max_exp = max_exp
        self.cached_dir = cached_dir
        self.memory_buffer = memory_buffer
        self.bin_prefix = bin_prefix
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.num_timesteps = num_timesteps
        
        self.shuffle = shuffle
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.log_precursor = log_precursor
        
        self.load()
        self.on_epoch_end()
        
        if any([self.scaler_x is None, self.scaler_y is None]):
            self.fit_scalers()
            
    def load(self):
        file_list = glob.glob(os.path.join(self.path, 'ML2019_*'))
        self.file_list = sorted(file_list, key = lambda x: int(x.split("Exp")[1].strip(".csv")))
        self.summary_file = pd.read_csv(
            os.path.join(self.path, self.summary_file), skiprows = 3
        )
        self.summary_file.columns = [x.strip() for x in self.summary_file.columns]
        
        # Set up a memory buffer if told to
        if self.memory_buffer is not None:
            self.memory_buffer = {}
        
        # Check if a cache dir was given
        if self.cached_dir is not None:
            if not os.path.isdir(self.cached_dir):
                raise OSError(
                    "The cached directory was specified but does not exist. Mkdir and try again."
                )

    def get_transform(self):
        return self.scaler_x, self.scaler_y

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.experiment_subset)
    
    def __getitem__(self, idx, fit = False):
        'Generate one data point'
        
        ### Find the relevant experiment file 
        exp = self.experiment_subset[idx]
        
        ### Do the shuffle, if we asked for that after each epoch,  here
        ### to ensure that it gets carried out
        
        self.processed += 1
        if self.processed == self.__len__():
            self.on_epoch_end()
        
        ### First see if we have it in memory
        if isinstance(self.memory_buffer, dict) and not fit:
            if idx in self.memory_buffer:
                return self.memory_buffer[idx]
            
        ### If experiment not in memory, check if saving pre-processed data to file
        if os.path.isdir(self.cached_dir) and not fit:
            ### If we have this experiment already cached, load it
            cached = f"{self.cached_dir}/{exp}.pkl"
            if os.path.isfile(cached) and self.scaler_x is not None:
                with open(cached, "rb") as fid:
                    return pickle.load(fid)
        
        ### Otherwise we have to create it for the first time.
        
        ### First find the relevant file for the selected experiment
        for file_path in self.file_list:
            if f"Exp{exp}" in file_path:
                break

        ### Next load the experiment file
        df = pd.read_csv(file_path)
        df.columns = [x.strip() for x in df.columns]

        ### Then load the summary file
        exp_num = int(re.findall("_Exp(\d+)*", file_path)[0])
        summary_file = self.summary_file[(
            self.summary_file['id'] == f"Exp{exp_num}"
        )].copy()

        for variable in summary_file.columns:
            df[variable] = summary_file[variable][exp_num]

        if len(self.bin_prefix) > 0: 
            for prefix in self.bin_prefix:
                df[prefix] = df.loc[:, df.columns.str.contains(prefix, regex=False)].sum(axis=1)

        ### Add the diurnal signal
        df = self.add_diurnal_signal(df)        
                
        ### Make the input/output arrays
        input_subset = df[self.input_vars].iloc[:-1,:].copy()
        output_subset = df[self.output_vars].iloc[1:,:].copy()
        
        ### Log the precursor [optional]
        if self.log_precursor:
            input_subset['Precursor [ug/m3]'] = np.log(input_subset['Precursor [ug/m3]'])
            output_subset['Precursor [ug/m3]'] = np.log(output_subset['Precursor [ug/m3]'])
        
        if "index" in input_subset.columns:
            input_subset = input_subset.drop(columns = ["index"])
        if "index" in output_subset.columns:
            output_subset = output_subset.drop(columns = ["index"])

        if self.scaler_x is not None: # Gets skipped when fitting the scalers
            input_subset = self.scaler_x.transform(
                input_subset.drop(['Time [s]', 'id'], axis=1)
            )
        if self.scaler_y is not None: # Gets skipped when fitting the scalers
            output_subset = self.scaler_y.transform(
                output_subset.drop(['Time [s]', 'id'], axis=1)
            )

        if fit:
            return input_subset, output_subset

        reshaped_input = torch.from_numpy(input_subset.astype(np.float32))
        reshaped_output = torch.from_numpy(output_subset.astype(np.float32))
            
        # Serialize the data and save
        if os.path.isdir(self.cached_dir) and not fit:
            cached = f"{self.cached_dir}/{exp}.pkl"
            with open(cached, "wb") as fid:
                pickle.dump((reshaped_input, reshaped_output), fid)
                
        # Save the data to the memory buffer
        if isinstance(self.memory_buffer, dict) and not fit:
            self.memory_buffer[idx] = (reshaped_input, reshaped_output)
                
        #if fit:
        #    return input_subset, output_subset
        
        return reshaped_input, reshaped_output
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.processed = 0
        if self.shuffle == True:
            random.shuffle(self.experiment_subset)
            
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
        
        if os.path.isfile(filepath):
            logger.info(
                f"Loading fitted data transformation models from {filepath}"
            )
            with open(filepath, "rb") as fid:
                self.num_timesteps, self.scaler_x, self.scaler_y = pickle.load(fid)
            
        else:
            logger.info(
                "Fitting data transformation models: MinMaxScaler((0, 1))"
            )
            worker = partial(self.__getitem__, fit = True)
            with Pool(8) as p:
                xs, ys = zip(*[(x,y) for (x,y) in tqdm.tqdm(
                    p.imap(worker, range(len(self.experiment_subset))),
                    total = len(self.experiment_subset))
                ])
            p.join()
            p.close()                    
                    
            xs = pd.concat(xs)
            ys = pd.concat(ys)
            self.reshape = True
                        
            self.num_timesteps = xs['Time [s]'].nunique()

            self.scaler_x = MinMaxScaler((0, 1)) #Pipeline(
                #steps=[('quant', QuantileTransformer()), ('minmax', MinMaxScaler((0, 1)))]
            #)
            self.scaler_y = MinMaxScaler((0, 1)) #Pipeline(
                #steps=[('quant', QuantileTransformer()), ('minmax', MinMaxScaler((0, 1)))]
            #)
            
            scaled_in_train = self.scaler_x.fit_transform(
                xs.drop(['Time [s]', 'id'], axis=1)
            )
            scaled_out_train = self.scaler_y.fit_transform(
                ys.drop(['Time [s]', 'id'], axis=1)
            )

            with open(filepath, "wb") as fid:
                pickle.dump([self.num_timesteps, self.scaler_x, self.scaler_y], fid)
            
            logger.info(
                f"Saved data transformation models to {filepath}"
            )

            
class BatchGeckoData:
    
    def __init__(self, split, data_path, 
                 input_cols, output_cols, 
                 shuffle = True, scaler_x = None, scaler_y = None):
        
        self.data_path = data_path
        self.split = split
        self.input_cols = input_cols
        self.output_cols = output_cols

        self.shuffle = shuffle
        
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        
        self.load()
        self.on_epoch_end()
        
    def load(self):
        self.x = pd.read_csv(os.path.join(self.data_path, f"x_{self.split}.csv"))
        self.y = pd.read_csv(os.path.join(self.data_path, f"y_{self.split}.csv"))        
        self.weights = self.y["weight"].copy().to_numpy(dtype=np.float32)
        
        self.x = self.x[self.input_cols].copy()
        self.y = self.y[self.output_cols].copy()
        
        self.x['Precursor [ug/m3]'] = np.log(self.x['Precursor [ug/m3]'])
        self.y['Precursor [ug/m3]'] = np.log(self.y['Precursor [ug/m3]'])
        
        drop_cols = ["index", "id", "indexer"]
        for df in [self.x, self.y]:
            keep_cols = [x for x in df.columns if x not in drop_cols]
            df = df[keep_cols].copy()
            
        if self.scaler_x is not None and self.scaler_y is not None:
            self.x = self.scaler_x.transform(
                self.x.drop(['Time [s]', 'id'], axis=1)
            )
            self.y = self.scaler_y.transform(
                self.y.drop(['Time [s]', 'id'], axis=1)
            )
        else:
            self.scaler_x = MinMaxScaler((0, 1))
            self.scaler_y = MinMaxScaler((0, 1))
            self.x = self.scaler_x.fit_transform(
                self.x.drop(['Time [s]', 'id'], axis=1)
            )
            self.y = self.scaler_y.fit_transform(
                self.y.drop(['Time [s]', 'id'], axis=1)
            )
            with open(os.path.join(self.data_path, "scalers.pkl"), "wb") as fid:
                pickle.dump([self.scaler_x, self.scaler_y], fid)
                
    def get_transform(self):
        return self.scaler_x, self.scaler_y
        
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        
        self.processed += 1
        if self.processed == self.__len__():
            self.on_epoch_end()
        
        x_data = torch.from_numpy(self.x[idx].astype(np.float32))
        y_data = torch.from_numpy(self.y[idx].astype(np.float32))
        w_data = np.array([self.weights[idx].astype(np.float32)])
        w_data = torch.from_numpy(w_data)
        
        return x_data, y_data, w_data
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.processed = 0
        if self.shuffle == True:
            pass