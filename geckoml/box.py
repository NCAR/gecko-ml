import joblib
import numpy as np
from tensorflow.keras.models import Model, load_model


class GeckoBoxEmulator(object):

    def __init__(self, neural_net_path, output_scaler_path):

        self.neural_net_path = neural_net_path
        self.output_scaler_path = output_scaler_path

        return

    def predict(self, input_concentrations, num_timesteps):

        output_scaler = joblib.load(self.output_scaler_path)
        mod = load_model(self.neural_net_path)


        static_input = input_concentrations[:, -6:]

        for i in range(num_timesteps):

            if i == 0:

                pred = mod.predict(input_concentrations)
                new_input = np.concatenate([pred, static_input], axis=1)
                pred_array = pred

            else:

                pred = mod.predict(new_input)
                new_input = np.concatenate([pred, static_input], axis=1)
                pred_array = np.concatenate([pred_array, pred], axis=0)

        results = output_scaler.inverse_transform(pred_array)

        return results