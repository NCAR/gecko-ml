import pandas as pd
import numpy as np
import random
import gc
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.framework.ops import disable_eager_execution
from .data import inverse_log_transform
import torch

from sklearn.metrics import mean_squared_error, mean_absolute_error
from geckoml.metrics import get_stability


disable_eager_execution()


class GeckoBoxEmulator(object):
    """
    Forward running box emulator for the GECKO-A atmospheric chemistry model. Uses first timestep of an experiment
    as the initial conditions for first prediction of neural network - uses that prediction as the new input for
    another prediction, and so on, looping through the length of an experiment.

    Attributes:
          neural_net_path (str): Path to saved neural network.
          output_scaler (str): Y Scaler object used on data to train the neural network.
          input_cols (list): Columns used as input into model
          output_cols (list): Columns used as output to model
          seed (int): Random seed
    """

    def __init__(self, neural_net_path, output_scaler, input_cols, output_cols, seed=8176):

        self.neural_net_path = neural_net_path
        self.output_scaler = output_scaler
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.seed = seed

        return

    def run_ensemble(self, client, data, num_timesteps, exps='all'):
        """
        Run an ensemble of GECKO-A experiment emulations distributed over a cluster using dask distributed.
        Args:
            client: Dask distributed TCP client
            data (numpy array): Scaled input Validation/testing dataframe, split by experiment.
            num_timesteps (int): Number of timesteps to run each emulation forward.
            exps (list of integers or 'all'): Number of experiments to run. Defaults to 'all' within data provided. If (int),
                    choose experiments randomly from those available.
        Returns:
            results_df (DataFrame): A concatenated pandas DataFrame of emulation results.
        """
        np.random.seed(self.seed)

        if exps == 'all':
            exps = data['id'].unique()
        else:
            exps = ['Exp' + str(i) for i in exps]

        starting_conds, temps, initial_out_values = [], [], []
        time_series = data[data['id'] == exps[0]]['Time [s]']

        for x in exps:
            data_sub = data[data['id'] == x].iloc[:, 1:-1].copy()
            data_sub.columns = self.input_cols[1:-1]
            temperature_ts = data_sub['temperature (K)'][1:].values
            sc = data_sub.iloc[0:1, :].values
            starting_conds.append(sc)
            temps.append(temperature_ts)

        futures = client.map(self.predict, starting_conds, [num_timesteps]*len(exps), temps,
                             [time_series]*len(exps), exps)
        results = client.gather(futures)
        results_df = pd.concat(results)

        return results_df

    def predict(self, starting_conds, num_timesteps, temps, time_series, exp):
        """ Run emulation of single experiment given initial conditions.
        Args:
            starting_conds (DataFrame): DataFrame of initial conditions.
            num_timesteps (int): Length of timesteps to run emulation.
            temps (Pandas Series): Timeseries of diurnally varying temperatures for respective experiment.
            time_series (Pandas Series): Pandas 'Time' column from experiment data.
            exp: Experiment number

        Returns:
            results (DataFrame): Pandas dataframe of emulated values with time stamps.
        """

        mod = load_model(self.neural_net_path)
        num_env_vars = len(self.input_cols) - len(self.output_cols)
        results = np.empty((num_timesteps, starting_conds.shape[-1] - num_env_vars))
        new_input = np.empty((1, starting_conds.shape[-1]))
        static_input = starting_conds[:, -num_env_vars:]

        for i in range(num_timesteps):

            if i == 0:

                pred = np.block(mod.predict(starting_conds))
                transformed_pred = self.output_scaler.inverse_transform(pred)
                results[i, :] = transformed_pred
                new_input[:, -num_env_vars:] = static_input
                new_input[:, :-num_env_vars] = pred
                new_input[:, 3] = temps[i]

            else:

                pred = np.block(mod.predict(new_input))
                transformed_pred = self.output_scaler.inverse_transform(pred)
                results[i, :] = transformed_pred

                if i < range(num_timesteps)[-1]:
                    new_input[:, -num_env_vars:] = static_input
                    new_input[:, :-num_env_vars] = pred
                    new_input[:, 3] = temps[i]

        results_df = pd.DataFrame(results)
        results_df.columns = self.output_cols[1:-1]
        results_df['Time [s]'] = time_series.values
        results_df['id'] = exp
        results_df = results_df.reindex(self.output_cols, axis=1)
        results_df = inverse_log_transform(results_df, ['Precursor [ug/m3]'])

        del mod
        tf.keras.backend.clear_session()
        gc.collect()

        return results_df

    @staticmethod
    def get_starting_conds(data, exp, seq_len=1, starting_ts=0):
        """
        Retrieve initial conditions for given experiment.
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

    def scale_input(self, input_concentrations, seq_length=1, starting_ts=0):
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
        num_env_vars = len(self.input_cols) - len(self.output_cols)
        scaled_input = self.input_scaler.transform(input_concentrations.iloc[starting_ts:seq_length, 1:-1])
        static_input = scaled_input[:, -num_env_vars:]

        return scaled_input, static_input


class GeckoBoxEmulatorTS(object):
    """
    Forward running box emulator for the GECKO-A atmospheric chemistry model. Uses first timestep of an experiment
    as the initial conditions for first prediction of neural network - uses that prediction as the new input for
    another prediction, and so on, looping through the length of an experiment.

    Attributes:
          neural_net_path (str): Path to saved neural network.
          output_scaler (str): Path to output scaler object used to train neural network.
          seq_length (int): Sequence length used to train RNN/LSTM network.
          input_cols (list): List of input variables.
          output_cols (lsit): List of output variables.
    """

    def __init__(self, neural_net_path, output_scaler, seq_length, input_cols, output_cols, seed=8176):

        self.neural_net_path = neural_net_path
        self.output_scaler = output_scaler
        self.seq_length = seq_length
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.seed = seed

        return

    def run_ensemble(self, client, data, num_timesteps, exps='all'):
        """
        Run an ensemble of GECKO-A experiment emulations distributed over a cluster using dask distributed.
        Args:
            client: Dask distributed TCP client.
            data (numpy array): Scaled Validation/testing data, split by experiment.
            num_timesteps (int): Number of timesteps to run each emulation forward.
            num_exps (int or 'all'): Number of experiments to run. Defaults to 'all' within data provided. If (int),
                    choose experiments randomly from those available.
        Returns:
            results_df (DataFrame): A concatenated pandas DataFrame of emulation results.
        """
        np.random.seed(self.seed)
        num_seq_ts = num_timesteps - self.seq_length + 1

        if exps == 'all':
            exps = data['id'].unique()
        else:
            exps = ['Exp' + str(i) for i in exps]

        starting_conds, temps, initial_out_values = [], [], []
        time_series = data[data['id'] == exps[0]].iloc[-num_seq_ts:, :]['Time [s]'].copy()

        for x in exps:
            data_sub = data[data['id'] == x].iloc[:, 1:-1].copy()
            data_sub.columns = self.input_cols[1:-1]
            sc = self.get_starting_conds_ts(data_sub)
            temperature_ts = data_sub['temperature (K)'][self.seq_length:].values
            starting_conds.append(sc)
            temps.append(temperature_ts)

        futures = client.map(self.predict_ts, starting_conds, [num_timesteps] * len(exps), temps,
                             [time_series] * len(exps), exps)
        results = client.gather(futures)
        results_df = pd.concat(results)
        results_df.columns = [str(x) for x in results_df.columns]

        return results_df

    def predict_ts(self, starting_conds, num_timesteps, temps, time_series, exp):
        """ Run emulation of single experiment given initial conditions.
        Args:
            starting_conds (DataFrame): DataFrame of initial conditions.
            num_timesteps (int): Length of time steps to run emulation.
            temps (Pandas Series): Time series of diurnally varying temperatures for respective experiment.
            time_series (Pandas Series): Pandas 'Time' column from experiment data.
            exp (Pandas Series): Series of Experiment ID

        Returns:
            results (DataFrame): Pandas dataframe of emulated values with time stamps.
        """
        num_env_vars = len(self.input_cols) - len(self.output_cols)
        mod = load_model(self.neural_net_path)
        ts = num_timesteps - self.seq_length + 1
        results = np.empty((ts, starting_conds.shape[2] - num_env_vars))
        new_input_single = np.empty((1, 1, starting_conds.shape[2]))
        static_input = starting_conds[0, 0, -num_env_vars:]

        for i in range(ts):

            if i == 0:

                pred = np.block(mod.predict(starting_conds))
                transformed_pred = self.output_scaler.inverse_transform(pred)
                results[i, :] = transformed_pred
                new_input_single[:, :, -num_env_vars:] = static_input
                new_input_single[:, :, :-num_env_vars] = pred
                new_input_single[:, :, 3] = temps[i]
                x = starting_conds[:, 1:, :]
                new_input = np.concatenate([x, new_input_single], axis=1)

            else:

                pred = np.block(mod.predict(new_input))
                transformed_pred = self.output_scaler.inverse_transform(pred)
                results[i, :] = transformed_pred

                if i < range(ts)[-1]:
                    new_input_single[:, :, -num_env_vars:] = static_input
                    new_input_single[:, :, :-num_env_vars] = pred
                    new_input_single[:, :, 3] = temps[i]
                    x = new_input[:, 1:, :]
                    new_input = np.concatenate([x, new_input_single], axis=1)

        results_df = pd.DataFrame(results)
        results_df.columns = self.output_cols[1:-1]
        results_df['Time [s]'] = time_series.values
        results_df['id'] = exp
        results_df = results_df.reindex(self.output_cols, axis=1)
        results_df = inverse_log_transform(results_df, ['Precursor [ug/m3]'])

        del mod
        tf.keras.backend.clear_session()
        gc.collect()

        return results_df

    def get_starting_conds_ts(self, data, starting_ts=0):
        """ Get 3D starting conditions in the format accepted by an RNN/LSTM Model.
        Args:
            data (Pandas DataFrame): Input Dataframe of specific experiment
            starting_ts (int): Timestep to pull initial conditions from
        Return:
            sc (numpy array): 3D array of sequenced data used for input into an RNN/LSTM model.
        """

        sc = np.zeros((1, self.seq_length, data.shape[1]))
        for i in range(self.seq_length):
            sc[0, i, :] = data.iloc[starting_ts + i, :].copy()

        return sc

    
### Add trainer classes for training and validation mode

def rnn_box_train_one_epoch(model, 
                            optimizer, 
                            loss_fn, 
                            batch_size, 
                            exps, 
                            num_timesteps, 
                            in_array, 
                            env_array, 
                            hidden_weight = 1.0, 
                            grad_clip = 1.0):
    
    """ Train an RNN model for one epoch given training data as input
    Args:
        model (torch.nn.Module)
        optimizer (torch.nn.Module)
        loss_fn (torch.nn.Module)
        batch_size (int)
        exps (List[str])
        num_timesteps (int)
        in_array (np.array)
        env_array (np.array)
    Return:
        train_loss (float)
        model (torch.nn.Module)
        optimizer (torch.nn.Module)
    """
    
    # Set the model to training mode
    model.train()
    
    # Grab the device from the model
    device = model._device()

    # Prepare the training dataset.
    num_experiments = in_array.shape[0]
    batches_per_epoch = int(np.ceil(num_experiments / batch_size))

    batched_experiments = list(range(batches_per_epoch))
    random.shuffle(batched_experiments)

    train_epoch_loss = []
    for j in batched_experiments:

        _in_array = torch.from_numpy(in_array[j * batch_size: (j + 1) * batch_size]).to(device).float()
        _env_array = torch.from_numpy(env_array[j * batch_size: (j + 1) * batch_size]).to(device).float()

        # Use initial condition @ t = 0 and get the first prediction
        # Clear gradient
        optimizer.zero_grad()

        h0 = model.init_hidden(_in_array[:, 0, :])
        pred, h0 = model(_in_array[:, 0, :], h0)

        # get loss for the predicted output
        loss = loss_fn(_in_array[:, 1, :3], pred)

        # get gradients w.r.t to parameters
        loss.backward()
        train_epoch_loss.append(loss.item())

        # update parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for i in range(1, num_timesteps-1): 
            # Use the last prediction to get the next prediction
            optimizer.zero_grad()
            # static envs 
            temperature = _in_array[:, i, 3:4]
            static_env = _env_array[:, -5:]
            new_input = torch.cat([pred.detach(), temperature, static_env], 1)
            
            # predict hidden state
            h0_pred = model.init_hidden(new_input.cpu())
            # compute loss for the last hidden prediction
            hidden_loss = loss_fn(h0.detach(), h0_pred)
            
            # predict next state with the GRU
            pred, h0 = model(new_input, h0.detach())
            loss = loss_fn(_in_array[:, i+1, :3], pred)
            
            # combine losses
            loss += hidden_weight * hidden_loss

            # get gradients w.r.t to parameters
            loss.backward()
            train_epoch_loss.append(loss.item())

            # update parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

    train_loss = np.mean(train_epoch_loss)
    
    return train_loss, model, optimizer


def rnn_box_test(model, 
                 exps, 
                 num_timesteps, 
                 in_array, 
                 env_array, 
                 y_scaler, 
                 output_cols, 
                 out_val, 
                 stable_thresh = 1.0, 
                 start_times = [0]):
    
    """ Run an RNN model in inference mode data as input
    Args:
        model (torch.nn.Module)
        exps (List[str])
        num_timesteps (int)
        in_array (np.array)
        env_array (np.array)
        y_scaler (sklearn.preprocessing._data.Scaler)
        output_cols (List[str])
        out_val (pd.DataFrame)
        stable_thresh (float)
        starting_time (int)
    Return:
        box_mae (float)
        scaled_box_mae (float)
        preds (pd.DataFrame)
        truth (pd.DataFrame)
    """
    
    epoch_loss = []
    box_loss_mae = []
    
    for start_time in start_times:
        
        with torch.no_grad():
            # use initial condition @ t = 0 and get the first prediction
            pred_array = np.empty((len(exps), 1439-start_time, 3))
            h0 = model.init_hidden(torch.from_numpy(in_array[:, start_time, :]).float())
            pred, h0 = model.predict(in_array[:, start_time, :], h0)
            pred_array[:, 0, :] = pred
            loss = mean_absolute_error(in_array[:, start_time + 1, :3], pred)
            epoch_loss.append(loss)

            # use the first prediction to get the next, and so on for num_timesteps
            for k, i in enumerate(range(start_time + 1, num_timesteps)): 
                temperature = in_array[:, i, 3:4]
                static_env = env_array[:, -5:]
                new_input = np.block([pred, temperature, static_env])
                pred, h0 = model.predict(new_input, h0)
                pred_array[:, k+1, :] = pred
                if i < (num_timesteps-1):
                    loss = mean_absolute_error(in_array[:, i+1, :3], pred)
                    epoch_loss.append(loss)

        # loop over the batch to fill up results dict
        results_dict = {}
        for k, exp in enumerate(exps):
            results_dict[exp] = pd.DataFrame(y_scaler.inverse_transform(pred_array[k]), columns=output_cols[1:-1])
            results_dict[exp]['id'] = exp
            results_dict[exp]['Time [s]'] = out_val['Time [s]'].unique()[start_time:]
            results_dict[exp] = results_dict[exp].reindex(output_cols, axis=1)

        preds = pd.concat(results_dict.values())
        truth = out_val.loc[out_val['id'].isin(exps)]
        truth = truth.sort_values(['id', 'Time [s]']).reset_index(drop=True)
        preds = preds.sort_values(['id', 'Time [s]']).reset_index(drop=True)

        start_time_cond = truth['Time [s]'].isin(out_val['Time [s]'].unique()[start_time:])
        truth = truth[start_time_cond]

        # Check for instabilities
        preds = preds.copy()
        preds['Precursor [ug/m3]'] = 10**(preds['Precursor [ug/m3]'])
        truth['Precursor [ug/m3]'] = 10**(truth['Precursor [ug/m3]'])
        unstable = preds.groupby('id')['Precursor [ug/m3]'].apply(
            lambda x: x[(x > stable_thresh) | (x < -stable_thresh)].any())
        stable_exps = unstable[unstable == False].index
        failed_exps = unstable[unstable == True].index
        c1 = ~truth["id"].isin(failed_exps)
        c2 = ~preds["id"].isin(failed_exps)
        
        if c2.sum() == 0:
            box_mae = 1.0
        else:
            box_mae = mean_absolute_error(preds[c2].iloc[:, 1:-1], truth[c1].iloc[:, 1:-1])
        box_loss_mae.append(box_mae)
        
    box_mae = np.mean(box_loss_mae)
    scaled_box_mae = np.mean(epoch_loss)
    
    return scaled_box_mae, box_mae, preds, truth