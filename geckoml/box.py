import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()


class GeckoBoxEmulator(object):
    """
    Forward running box emulator for the GECKO-A atmospheric chemistry model. Uses first timestep of an experiment
    as the initial conditions for first prediction of neural network - uses that prediction as the new input for
    another prediction, and so on, looping through the length of an experiment.

    Attributes:
          neural_net_path (str): Path to saved neural network.
          input_scaler (str): X Scaler object used on data to train the neural network.
          output_scaler (str): Y Scaler object used on data to train the neural network.
          input_cols (list): Columns used as input into model
          output_cols (list): Columns used as output to model
          seed (int): Random seed
    """

    def __init__(self, neural_net_path, input_cols, output_cols):

        self.neural_net_path = neural_net_path
        self.input_cols = input_cols
        self.output_cols = output_cols

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
        init_array = batched_array[:, 0, :]
        pred_array = np.empty((n_exps, n_timesteps, len(self.output_cols)))
        mod = load_model(self.neural_net_path)

        for time_step in range(n_timesteps):

            if time_step == 0:
                pred = np.block(mod.predict(init_array))
            else:
                pred = np.block(mod.predict(new_input))
            new_input = batched_array[:, time_step, :]
            new_input[:, out_col_idx] = pred
            pred_array[:, time_step, :] = pred

        preds_df = pd.DataFrame(data=pred_array.reshape(-1, len(self.output_cols)),
                                columns=raw_val_output.columns, index=raw_val_output.index)
        return truth, preds_df
