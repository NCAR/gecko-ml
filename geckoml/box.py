import joblib
import numpy as np
from tensorflow.keras.models import Model, load_model


class GeckoBoxEmulator(object):

    def __init__(self, neural_net_path, input_scaler_path, output_scaler_path):

        self.neural_net_path = neural_net_path
        self.input_scaler_path = input_scaler_path
        self.output_scaler_path = output_scaler_path

        return

    def run_ensemble(self, data, num_exps, num_exps='all'):
        """ if num_exps == 'all' then... """
        exps = data['id'].unique()

        if num_exps != 'all':

            exps = np.random.choice(exps, num_exps, replace=False)

        for x in exps:

            sc = self.get_starting_conds(data, x)
            num_timesteps = len(data)
            pred = self.predict(sc, num_timesteps)

            # append preds/results of each experimetn

        return result_df

    def predict(self, input_concentrations, num_timesteps, starting_ts=0, seq_length=1):

        input_scaler = joblib.load(self.input_scaler_path)
        output_scaler = joblib.load(self.output_scaler_path)
        mod = load_model(self.neural_net_path)

        scaled_input = input_scaler.transform(input_concentrations.iloc[starting_ts:seq_length,1:-1])
        static_input = scaled_input[:, -6:]

        for i in range(num_timesteps):

            if i == 0:

                pred = mod.predict(scaled_input)
                new_input = np.concatenate([pred, static_input], axis=1)
                pred_array = pred

            else:

                pred = mod.predict(new_input)
                new_input = np.concatenate([pred, static_input], axis=1)
                pred_array = np.concatenate([pred_array, pred], axis=0)

        results = output_scaler.inverse_transform(pred_array)

        return results

    def get_starting_conds(self, data, exp, seq_len=1, starting_ts=0):
        starting_conditions = data[data['id'] == exp].iloc[starting_ts:starting_ts + seq_len, :]
        return starting_conditions

    def scale_input(self, input_concentrations):

        input_scaler = joblib.load(self.input_scaler_path)
        scaled_input = input_scaler.transform(input_concentrations.iloc[starting_ts:seq_length,1:-1])
        static_input = scaled_input[:, -6:]

        return scaled_input, static_input

    def transform_output(self, predictions):
        return

