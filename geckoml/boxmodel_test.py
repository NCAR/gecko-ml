import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from geckoml.models import DenseNeuralNetwork
from geckoml.box import GeckoBoxEmulator
import pandas as pd
import numpy as np
import yaml 
import os
import gc
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.framework.ops import disable_eager_execution
from geckoml.data import partition_y_output, get_output_scaler, save_metrics
import os.path
from os import path

np.random.seed(9)

#Call from gecko_ml

#Test box model
def test_boxmodel():

    config_file= "config/dodecane.yml"
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
    seq_length = config['seq_length']
    train_start_exp = config['train_start_exp']
    train_end_exp = config['train_end_exp']
    val_start_exp = config['val_start_exp']
    val_end_exp = config['val_end_exp']
    test_start_exp = config['test_start_exp']
    test_end_exp = config['test_end_exp']
    box_val_exps = config['box_val_exps']

    assert len(input_vars) == 37
    assert len(output_vars) == 31
    
    # Unit Test

    in_train = pd.read_csv("test_data/in_train_test.csv")
    out_train = pd.read_csv("test_data/out_train_test.csv")

    # Rescale training and validation / testing data
    x_scaler = MinMaxScaler()
    scaled_in_train = x_scaler.fit_transform(in_train.drop(['Time [s]', 'id'], axis=1))
    
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
    model.build_neural_network(scaled_in_train.shape[1],[scaled_out_train.shape[1]])
    result = model.model.fit(scaled_in_train, scaled_out_train)

    # Save model and scaler
    model.model.save("test_data/test.h5")
    test_path = "test_data/test.h5"
    
    assert os.path.exists(test_path)
    
    mod = GeckoBoxEmulator(
        neural_net_path = test_path, 
        output_scaler=y_scaler,
        input_cols=input_vars, 
        output_cols=output_vars
    )

    
    starting_conds = scaled_in_train[0].reshape((1, scaled_in_train[0].shape[0]))
    temps = in_train['temperature (K)']
    time_series = in_train['Time [s]']
    exp = 'Exp0'
    results = mod.predict(
        starting_conds,
        num_timesteps,
        temps,
        time_series,
        exp
    )
    
    assert len(time_series) == 1439
    assert starting_conds.shape == (1, 35)
    assert results.shape[0] == 1439
    assert isinstance(results, pd.DataFrame)

    if os.path.isfile("test_data/test.h5"):
        os.remove("test_data/test.h5")
        
    assert not os.path.exists(test_path)
   
