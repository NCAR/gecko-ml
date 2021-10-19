import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.python.framework.ops import disable_eager_execution
import random
import torch

from .data import inv_transform_preds
from .metrics import ensembled_metrics

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

        pred = None
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


def rnn_box_train_one_epoch(model,
                            optimizer,
                            loss_fn,
                            batch_size,
                            in_array,
                            out_col_idx,
                            hidden_weight=1.0,
                            loss_weights=[1.0, 1.0, 1.0],
                            grad_clip=1.0):
    """ Train an RNN model for one epoch given training data as input
    Args:
        model (torch.nn.Module)
        optimizer (torch.nn.Module)
        loss_fn (torch.nn.Module)
        batch_size (int)
        in_array (np.array)
        out_col_idx (list)
        hidden_weight (float)
        loss_weights (list(float))
        grad_clip (float)
    Return:
        train_loss (float)
        model (torch.nn.Module)
        optimizer (torch.nn.Module)
    """

    # Set the model to training mode
    model.train()

    # Grab the device from the model
    device = model._device()

    # Move the weights to the device
    loss_weights = torch.FloatTensor(loss_weights).to(device)

    # Prepare the training dataset.
    num_timesteps = in_array.shape[1]
    num_experiments = in_array.shape[0]
    batches_per_epoch = int(np.ceil(num_experiments / batch_size))
    batched_experiments = list(range(batches_per_epoch))
    random.shuffle(batched_experiments)

    train_epoch_loss = []
    for j in batched_experiments:

        random_indices = list(range(in_array.shape[0]))
        random_selection = random.sample(random_indices, batch_size)
        _in_array = torch.from_numpy(in_array[random_selection]).to(device).float()

        # Use initial condition @ t = 0 and get the first prediction
        # Clear gradient
        optimizer.zero_grad()

        h0 = model.init_hidden(_in_array[:, 0, :])
        pred, h0 = model(_in_array[:, 0, :], h0)

        # get loss for the predicted output
        loss = loss_fn(_in_array[:, 1, out_col_idx], pred)
        loss = (loss_weights * loss).mean()

        # get gradients w.r.t to parameters
        loss.backward()
        train_epoch_loss.append(loss.item())

        # update parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for i in range(1, num_timesteps - 1):
            # Use the last prediction to get the next prediction
            optimizer.zero_grad()

            # update the next input to the model
            new_input = _in_array[:, i, :]
            new_input[:, out_col_idx] = pred.detach()

            # predict hidden state
            h0_pred = model.init_hidden(new_input.cpu())
            # compute loss for the last hidden prediction
            hidden_loss = loss_fn(h0.detach(), h0_pred).mean()
            # predict next state with the GRU
            pred, h0 = model(new_input, h0.detach())

            # get loss for the predicted output
            loss = loss_fn(_in_array[:, i + 1, out_col_idx], pred)
            loss = (loss_weights * loss).mean()

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
                 loss_fn,
                 in_array,
                 out_array,
                 y_scaler,
                 output_cols,
                 out_col_idx,
                 log_trans_cols,
                 tendency_cols,
                 stable_thresh=10,
                 start_times=[0]):
    """ Run an RNN model in inference mode data as input
    Args:
        model (torch.nn.Module)
        loss_fn (torch.nn.Module)
        in_array (np.array)
        out_array (pd.DataFrame)
        y_scaler (sklearn.preprocessing._data.Scaler)
        output_cols (List[str])
        out_col_idx (List[str])
        log_trans_cols (List[str])
        tendency_cols (List[str])
        stable_thresh (float)
        starting_times (List[int])
    Return:
        mean_step_loss (float)
        mean_box_mae (float)
        preds (pd.DataFrame)
        truth (pd.DataFrame)
    """

    # Put the model into eval model
    model.eval()

    # Grab the device from the model
    device = model._device()

    # How many total timesteps in the data
    num_timesteps = in_array.shape[1]

    all_preds, all_truths, total_loss = [], [], []
    for start_time in start_times:

        val_loss = []
        _in_array = torch.from_numpy(in_array).to(device).float()

        with torch.no_grad():

            # set up array for saving predicted results
            pred_array = np.empty((in_array.shape[0], num_timesteps - start_time, len(out_col_idx)))

            # use initial condition @ t = start_time and get the first prediction
            h0 = model.init_hidden(_in_array[:, start_time, :])
            pred, h0 = model(_in_array[:, start_time, :], h0)
            pred_array[:, 0, :] = pred.cpu().numpy()
            loss = loss_fn(_in_array[:, start_time + 1, out_col_idx], pred).item()
            val_loss.append(loss)

            # use the first prediction to get the next, and so on for num_timesteps
            for k, i in enumerate(range(start_time + 1, num_timesteps)):
                new_input = _in_array[:, i, :]
                new_input[:, out_col_idx] = pred
                pred, h0 = model(new_input, h0)
                pred_array[:, k + 1, :] = pred.cpu().numpy()
                if i < (num_timesteps - 1):
                    loss = loss_fn(_in_array[:, i + 1, out_col_idx], pred).item()
                    val_loss.append(loss)

        # put results into pandas df, first select indices relevant to the start time
        idx = out_array.index
        start_time_units = sorted(list(set([x[0] for x in idx])))[start_time]
        start_time_condition = [x[0] >= start_time_units for x in idx]
        idx = out_array[start_time_condition].index

        raw_box_preds = pd.DataFrame(
            data=pred_array.reshape(-1, len(output_cols)),
            columns=output_cols,
            index=idx
        )

        # inverse transform 
        truth, preds = inv_transform_preds(
            raw_preds=raw_box_preds,
            truth=out_array[start_time_condition],
            y_scaler=y_scaler,
            log_trans_cols=log_trans_cols,
            tendency_cols=tendency_cols)

        # Accumulate the results for each box simulation for different starting times
        all_preds.append(preds)
        all_truths.append(truth)

        # Accumulate the step losses across different starting times
        total_loss += val_loss

    all_preds = pd.concat(all_preds)
    all_truths = pd.concat(all_truths)
    metrics = ensembled_metrics(y_true=all_truths,
                                y_pred=all_preds,
                                member=0,
                                output_vars=output_cols,
                                stability_thresh=stable_thresh)
    mean_box_mae = metrics['mean_mae'].mean()
    mean_step_loss = np.sum(total_loss)

    return mean_step_loss, mean_box_mae, metrics, all_preds, all_truths
