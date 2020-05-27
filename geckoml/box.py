class GeckoBoxEmulator(object):
    def __init__(self, neural_net_path, input_scaler_path, output_scaler_path,
                 input_scaler_type, output_scaler_type,
                 temperature, zenith, aerosols, o3,
                 nox, oh):
        self.neural_net_path = neural_net_path
        self.input_scaler_path = input_scaler_path
        self.output_scaler_path = output_scaler_path
        self.input_scaler_type = input_scaler_type
        self.output_scaler_type = output_scaler_type
        self.temperature = temperature
        self.zenith = zenith
        self.aerosols = aerosols
        self.o3 = o3
        self.nox = nox
        self.oh = oh
        return

    def predict(self, input_concentrations, num_timesteps):
        return