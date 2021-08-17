import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.python.framework.ops import disable_eager_execution
import random
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

disable_eager_execution()


class GeckoBoxEmulator(object):
    """
    Model class to run Box model through time.
    Args:
        neural_net_path: Path to saved model
        input_cols: Feature names for input to model
        output_cols: Feature names for output of model
        model_object: Model object to be used in hyperparamter optimization. Deafults to None.
        hyper_opt: Whether or not this is being used in a hyperparameter tuning setting. Defaults to False.
    """

    def __init__(self, neural_net_path, input_cols, output_cols, model_object=None, hyper_opt=False):

        self.neural_net_path = neural_net_path
        self.input_cols = input_cols
        self.output_cols = output_cols
        if hyper_opt:
            self.mod = model_object
        else:
            self.mod = load_model(self.neural_net_path)

        return

    def run_box_simulation(self, raw_val_output, transformed_val_input, exps='all'):
        """
        Run an ensemble of GECKO-A experiment emulations distributed over a cluster using dask distributed.
        Args:
            raw_val_output( pd.DataFrame): Raw validation output dataframe
            transformed_val_input (pd.DataFrame): Transformed/scaled validation dataframe, split by experiment.
            exps (list of experiments or 'all'):
        Returns:
            results_df (DataFrame): A concatenated pandas DataFrame of emulation results.
        """
        truth = raw_val_output.copy()
        if exps != 'all':
            data_sub = transformed_val_input.loc[transformed_val_input.index.isin(exps, level='id')]
            truth = truth.loc[truth.index.isin(exps, level='id')]
        else:
            data_sub = transformed_val_input.copy(deep=True)

        n_exps = len(data_sub.index.unique(level='id'))
        n_timesteps = len(data_sub.index.unique(level='Time [s]'))
        n_features = len(self.input_cols)
        out_col_idx = data_sub.columns.get_indexer(self.output_cols)
        batched_array = data_sub.values.reshape(n_exps, n_timesteps, n_features)
        pred_array = np.empty((n_exps, n_timesteps, len(self.output_cols)))

        for time_step in range(n_timesteps):
            if time_step == 0:
                new_input = batched_array[:, time_step, :]
            elif time_step > 0:
                new_input = batched_array[:, time_step, :]
                new_input[:, out_col_idx] = pred

            pred = np.block(self.mod.predict(new_input))
            pred_array[:, time_step, :] = pred

        idx = raw_val_output.index

        preds_df = pd.DataFrame(data=pred_array.reshape(-1, len(self.output_cols)),
                                columns=self.output_cols, index=idx)
        return preds_df

    
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
