import joblib
import pandas as pd
import numpy as np
import gc
import tensorflow as tf
from tensorflow.keras.models import load_model
from dask.distributed import Client, LocalCluster


class GeckoBoxEmulator(object):
    """
    Forward running box emulator for the GECKO-A atmospheric chemistry model. Uses first timestep of an experiment
    as the initial conditions for first prediction of neural network - uses that prediction as the new input for
    another prediction, and so on, looping through the length of an experiment.

    Attributes:
          neural_net_path (str): Path to saved neural network.
          input_scaler_path (str): Path to input scaler object used to train neural network.
          output_scaler_path (str): Path to output scaler object used to train neural network.
    """
    def __init__(self, neural_net_path, input_scaler_path, output_scaler_path):

        self.neural_net_path = neural_net_path
        self.input_scaler_path = input_scaler_path
        self.output_scaler_path = output_scaler_path

        return

    def run_ensemble(self, data, num_timesteps, num_exps='all'):
        """
        Run an ensemble of GECKO-A experiment emulations distributed over a cluster using dask distributed.
        Args:
            data (DataFrame): Validation/testing dataframe split by experiment.
            num_timesteps (int): Number of timesteps to run each emulation forward.
            num_exps (int or 'all'): Number of experiments to run. Defaults to 'all' within data provided. If (int),
                    choose experiments randomly from those available.
        Returns:
            results_df (DataFrame): A concatenated pandas DataFrame of emulation results.
        """
        exps = data['id'].unique()
        if num_exps != 'all':
            exps = np.random.choice(exps, num_exps, replace=False)

        starting_conds = []
        time_series = data[data['id'] == exps[0]].iloc[1:num_timesteps+1, 0].reset_index(drop=True)
        for x in exps:
            sc = self.get_starting_conds(data, x)
            starting_conds.append(sc)

        cluster = LocalCluster(processes=True)
        client = Client(cluster)
        futures = client.map(self.predict, starting_conds, [num_timesteps]*len(exps), [time_series]*len(exps))
        results = client.gather(futures)
        results_df = pd.concat(results)
        client.shutdown()

        return results_df

    def predict(self, starting_conds, num_timesteps, time_series, starting_ts=0, seq_length=1):
        """ Run emulation of single experiment given initial conditions.
        Args:
            starting_conds (DataFrame): DataFrame of initial conditions.
            num_timesteps (int): Length of timesteps to run emulation.
            time_series (Pandas Series): Pandas 'Time' column from experiment data.
            starting_ts (int): Timestep number to start emulation (should match starting conditions). Defaults to 0.
            seq_length (int): Number of timesteps to use for single prediction. Must match the data shape used to
                    train the nueral network. Most RNN/LSTM will be > 1. Defaults to 0.

        Returns:
            results (DataFrame): Pandas dataframe of emulated values with time stamps.
        """
        tf.keras.backend.clear_session()
        gc.collect()
        input_scaler = joblib.load(self.input_scaler_path)
        output_scaler = joblib.load(self.output_scaler_path)
        mod = load_model(self.neural_net_path)

        scaled_input = input_scaler.transform(starting_conds.iloc[starting_ts:seq_length, 1:-1])
        static_input = scaled_input[:, -6:]
        exp = starting_conds['id'].values[0]

        for i in range(num_timesteps):

            if i == 0:

                pred = mod.predict(scaled_input)
                new_input = np.concatenate([pred, static_input], axis=1)
                pred_array = pred

            else:

                pred = mod.predict(new_input)
                new_input = np.concatenate([pred, static_input], axis=1)
                pred_array = np.concatenate([pred_array, pred], axis=0)

        results = pd.DataFrame(output_scaler.inverse_transform(pred_array))
        results.columns = starting_conds.columns[1:-7]
        results['id'] = exp
        results['Time [s]'] = time_series
        results = results.reset_index(drop=True)

        return results

    def get_starting_conds(self, data, exp, seq_len=1, starting_ts=0):
        """
        Retrive initial conditions for given experiment.
        Args:
             data (DataFrame): Validation or testing DataFrame.
             exp (str): Experiment label in form ('Exp####') without preceding zeros (ex. experiment 18 would be
                    'Exp18' not 'Exp0018')
            seq_len (int): Number of observations to use as initial conditions (> 1 for RNN/LSTM models) Defaults to 1.
            starting_ts (int): First timestep to use for initial conditions.
        Returns:
            starting_conditions (Dataframe): Dataframe of obs to use as input for first prediction of box emulator.

        """
        starting_conditions = data[data['id'] == exp].iloc[starting_ts:starting_ts + seq_len, :]
        return starting_conditions

    def scale_input(self, input_concentrations, seq_length=1, starting_ts=0, ):
        """
        Scale initial conditions for initial prediction of box emulator.
        Args:
            input_concentrations (DataFrame): DataFrame of initial conditions.
            seq_length (int): Number of obs to use as initial conditions (> 1 for RNN/LSTM models) Defaults to 1.
            starting_ts (int): First timestep to use for initial conditions.
        Returns:
            scaled_input (numpy array): scaled input array ready for predition.
            static_input (numpy array): scaled subset (env. conds) of input that remains static throughout emulation.
        """

        input_scaler = joblib.load(self.input_scaler_path)
        scaled_input = input_scaler.transform(input_concentrations.iloc[starting_ts:seq_length, 1:-1])
        static_input = scaled_input[:, -6:]

        return scaled_input, static_input


    def transform_output(self, predictions):
        return

