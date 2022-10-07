import warnings

warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from geckoml.models import DenseNeuralNetwork
from geckoml.box import GeckoBoxEmulator
import pandas as pd
import numpy as np
import yaml
import os
import os.path

np.random.seed(9)


# Call from gecko_ml

# Test box model
def test_boxmodel():
    config_file = "config/dodecane.yml"
    with open(config_file) as fid:
        config = yaml.load(fid)

    # Extract config arguments and validate if necessary
    species = config['species']
    dir_path = config['dir_path']
    aggregate_bins = config['aggregate_bins']
    bin_prefix = config['bin_prefix']
    input_vars = config['input_vars']
    output_vars = config['output_vars']
    output_path = config['output_path']
    scaler_type = config['scaler_type']

    assert len(input_vars) == 35
    assert len(output_vars) == 28

    # Unit Test

    in_train = pd.read_csv("./test_data/in_train_test.csv").set_index('id')
    out_train = pd.read_csv("./test_data/out_train_test.csv")

    # Rescale training and validation / testing data
    x_scaler = MinMaxScaler()
    scaled_in_train = x_scaler.fit_transform(in_train.drop(['Time [s]'], axis=1))

    # Load transform scaler
    y_scaler = MinMaxScaler()

    # Transform the data
    scaled_out_train = y_scaler.fit_transform(out_train.drop(['Time [s]', 'id'], axis=1))

    assert (0 <= scaled_in_train.all() <= 1)
    assert (0 <= scaled_out_train.all() <= 1)
    assert len(scaled_in_train) == 1439
    assert len(scaled_out_train) == 1439

    # Grab the time-steps
    num_timesteps = in_train['Time [s]'].nunique()

    assert num_timesteps == 1439

    # Train for 1 epoch on fake data 
    model = DenseNeuralNetwork(loss_weights={})
    model.build_neural_network(scaled_in_train.shape[1], [scaled_out_train.shape[1]])
    result = model.model.fit(scaled_in_train, scaled_out_train)

    # Save model and scaler
    model.model.save("test_data/test.h5")
    test_path = "test_data/test.h5"

    assert os.path.exists(test_path)

    mod = GeckoBoxEmulator(
        neural_net_path=test_path,
        input_cols=input_vars,
        output_cols=output_vars
    )
    scaled_train_copy = in_train.drop(['Time [s]'], axis=1).copy()
    scaled_train_copy.loc[:] = x_scaler.fit_transform(scaled_train_copy)
    time_series = in_train['Time [s]']
    exp = 'Exp0'
    results = mod.run_box_simulation(raw_val_output=in_train,
                                     transformed_val_input=scaled_train_copy,
                                     exps=exp)

    assert len(time_series) == 1439
    assert results.shape[0] == 1439
    assert isinstance(results, pd.DataFrame)

    if os.path.isfile("test_data/test.h5"):
        os.remove("test_data/test.h5")

    assert not os.path.exists(test_path)
