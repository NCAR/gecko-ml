import joblib
import pandas as pd
import numpy as np
import gc
import tensorflow as tf
from tensorflow.keras.models import load_model
from dask.distributed import Client, LocalCluster, wait

class GeckoBoxEmulator(object):

    def __init__(self, neural_net_path, input_scaler_path, output_scaler_path):

        self.neural_net_path = neural_net_path
        self.input_scaler_path = input_scaler_path
        self.output_scaler_path = output_scaler_path

        return

    def run_ensemble(self, data, num_timesteps, num_exps='all'):
        """ """
        exps = data['id'].unique()
        if num_exps != 'all':
            exps = np.random.choice(exps, num_exps, replace=False)

        starting_conds = []
        time_series = data[data['id'] == exps[0]].iloc[1:num_timesteps+1, 0].reset_index(drop=True)
        for x in exps:
            sc = self.get_starting_conds(data, x)
            starting_conds.append(sc)

        cluster = LocalCluster(processes=True, n_workers=72, threads_per_worker=1)
        client = Client(cluster)
        print(cluster)
        futures = client.map(self.predict, starting_conds, [num_timesteps]*len(exps), [time_series]*len(exps))
        wait(futures)
        results = client.gather(futures)
        results_df = pd.concat(results)
        client.shutdown()

        return results_df

    def predict(self, starting_conds, num_timesteps, time_series, starting_ts=0, seq_length=1):
        tf.keras.backend.clear_session()
        gc.collect()
        input_scaler = joblib.load(self.input_scaler_path)
        output_scaler = joblib.load(self.output_scaler_path)
        mod = load_model(self.neural_net_path)

        scaled_input = input_scaler.transform(starting_conds.iloc[starting_ts:seq_length,1:-1])
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
        starting_conditions = data[data['id'] == exp].iloc[starting_ts:starting_ts + seq_len, :]
        return starting_conditions

    def scale_input(self, input_concentrations):

        input_scaler = joblib.load(self.input_scaler_path)
        scaled_input = input_scaler.transform(input_concentrations.iloc[starting_ts:seq_length,1:-1])
        static_input = scaled_input[:, -6:]

        return scaled_input, static_input

    def transform_output(self, predictions):
        return




