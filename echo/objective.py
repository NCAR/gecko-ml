import sys
#sys.path.insert(0, '/glade/work/cbecker/gecko-ml/')
import warnings
warnings.filterwarnings("ignore")
import copy
import optuna
import logging
import traceback
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from aimlutils.echo.src.base_objective import *
    from aimlutils.echo.src.pruners import KerasPruningCallback
except ModuleNotFoundError:
    from aimlutils.echo.hyper_opt.base_objective import *
    from aimlutils.echo.hyper_opt.utils import KerasPruningCallback
except:
    raise OSError("aimlutils does not seem to be installed, or is not on your python path. Exiting.")
    
from geckoml.models import DenseNeuralNetwork
from geckoml.data import *
from geckoml.box import *
from geckoml.metrics import *
logger = logging.getLogger(__name__)
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from geckoml.callbacks import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input, Dense, Dropout, GaussianNoise, Activation, \
    Concatenate, BatchNormalization, LSTM, Conv1D, AveragePooling1D, MaxPooling1D, LeakyReLU, PReLU, ELU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.backend as K
from keras_self_attention import SeqSelfAttention
import tensorflow as tf
import numpy as np
import xarray as xr
import pandas as pd

import tqdm


tf.keras.backend.set_floatx('float64')


def custom_updates(trial, conf):
    # Get list of hyperparameters from the config
    hyperparameters = conf["optuna"]["parameters"]

#     # Now update some via custom rules
#     precursor_weight = trial.suggest_loguniform(**hyperparameters["precursor_weight"]["settings"])
#     gas_weight = trial.suggest_loguniform(**hyperparameters["gas_weight"]["settings"])
#     aerosol_weight = trial.suggest_loguniform(**hyperparameters["aerosol_weight"]["settings"])

#     conf["dense_network"]["loss_weights"] = [precursor_weight, gas_weight, aerosol_weight]

    return conf


class Objective(BaseObjective):
    
    def __init__(self, config, metric="box_mae", device="cpu"):
        
        # Initialize the base class
        BaseObjective.__init__(self, config, metric, device)


    def train(self, trial, conf):

        conf = custom_updates(trial, conf)

        # Set up some globals
        tf.random.set_seed(5999)
        
        scalers = {"MinMaxScaler": MinMaxScaler,
                   "StandardScaler": StandardScaler}
        
        species = conf['species']
        dir_path = conf['dir_path']
        summary_file = conf['summary_file']
        aggregate_bins = conf['aggregate_bins']
        bin_prefix = conf['bin_prefix']
        input_vars = conf['input_vars']
        output_vars = conf['output_vars']
        scaler_type = conf['scaler_type']
        exps = conf['box_val_exps']
        output_cols = conf['output_vars']

        # Load the data
        # Load GECKO experiment data, split into ML inputs and outputs and persistence outputs
        #input_data, output_data = combine_data(dir_path, summary_file, aggregate_bins, bin_prefix,
        #                                       input_vars, output_vars, species)

        # Split into training, validation, testing subsets
        #in_train, out_train, in_val, out_val, in_test, out_test = split_data(
        #    input_data=input_data,
        #    output_data=output_data,
        #    train_start=conf['train_start_exp'],
        #    train_end=conf['train_end_exp'],
        #    val_start=conf['val_start_exp'],
        #    val_end=conf['val_end_exp'],
        #    test_start=conf['test_start_exp'],
        #    test_end=conf['test_end_exp'])

        in_train = pd.read_csv('/glade/scratch/cbecker/gecko_data/apinO3_train_in_agg.csv')
        out_train = pd.read_csv('/glade/scratch/cbecker/gecko_data/apinO3_train_out_agg.csv')
        in_val = pd.read_csv('/glade/scratch/cbecker/gecko_data/apinO3_val_in_agg.csv')
        out_val = pd.read_csv('/glade/scratch/cbecker/gecko_data/apinO3_val_out_agg.csv')

        num_timesteps = in_train['Time [s]'].nunique()

        # Rescale training and validation / testing data
        if scaler_type == "MinMaxScaler":
            x_scaler = scalers[scaler_type]((conf['min_scale_range'], conf['max_scale_range']))
        else:
            x_scaler = scalers[scaler_type]()
        scaled_in_train = x_scaler.fit_transform(in_train.drop(['Time [s]', 'id'], axis=1))
        scaled_in_val = x_scaler.transform(in_val.drop(['Time [s]', 'id'], axis=1))

        y_scaler = get_output_scaler(x_scaler, output_vars, scaler_type, data_range=(
            conf['min_scale_range'], conf['max_scale_range']))
        scaled_out_train = y_scaler.transform(out_train.drop(['Time [s]', 'id'], axis=1))
        scaled_out_val = y_scaler.transform(out_val.drop(['Time [s]', 'id'], axis=1))

        y = partition_y_output(scaled_out_train, conf["dense_network"]['output_layers'], aggregate_bins)
        y_val = partition_y_output(scaled_out_val, conf["dense_network"]['output_layers'], aggregate_bins)
        
        # Batch the experiments 
        if exps == 'all':
            exps = list(in_val['id'].unique())

        in_array = []
        env_array = []
        for exp in exps:
            in_data = x_scaler.transform(in_val[in_val['id'] == exp].iloc[:, 1:-1])
            env_conds = in_data[0, -6:]
            in_array.append(np.expand_dims(in_data, axis=0)) # shape goes from (num_timesteps, outputs) -> (1, num_timesteps, outputs)
            env_array.append(np.expand_dims(env_conds, axis=0))
        in_array = np.concatenate(in_array) # (num_experiments, num_timesteps, outputs)
        env_array = np.concatenate(env_array)
            
        # Load callbacks        
        box_callback = BoxValidationCallback(self.metric, exps, num_timesteps, in_array, 
                                             env_array, y_scaler, output_cols, out_val)
        
        pruning_callback = KerasPruningCallback(trial, self.metric, interval = 1)
        callbacks = [box_callback, pruning_callback]
        callbacks += get_callbacks(conf["callbacks"]) if "callbacks" in conf else []

        # Load the model
        mod = CustomModel(**conf["dense_network"])
        
        # Train the model 
        rate = 1.0 #conf["dense_network"]["custom_activation_decay_rate"]
        history = mod.fit(scaled_in_train, y, scaled_in_val, y_val, callbacks, rate = rate)
            
        # Return the best box mae from the training history
        results = {
            self.metric: history.history[self.metric][box_callback.best_epoch]
        }

        return results
    
            
def box_validate(mod, exps, num_timesteps, in_array, env_array, y_scaler, output_cols, out_val):
    
    # use initial condition @ t = 0 and get the first prediction
    pred_array = np.empty((len(exps), 1439, 3))
    pred = mod.predict(in_array[:, 0, :])
    pred_array[:, 0, :] = pred

    # use the first prediction to get the next, and so on for num_timesteps
    for i in range(1, num_timesteps):
        new_input = np.block([pred, env_array])
        pred = mod.predict(new_input)
        pred_array[:, i, :] = pred

    # loop over the batch to fill up results dict
    results_dict = {}
    for k, exp in enumerate(exps):
        results_dict[exp] = pd.DataFrame(y_scaler.inverse_transform(pred_array[k]), columns=output_cols[1:-1])
        results_dict[exp]['id'] = exp
        results_dict[exp]['Time [s]'] = out_val['Time [s]'].unique()
        results_dict[exp] = results_dict[exp].reindex(output_cols, axis=1)

    preds = pd.concat(results_dict.values())
    truth = out_val.loc[out_val['id'].isin(exps)]
    truth = truth.sort_values(['id', 'Time [s]']).reset_index(drop=True)
    preds = preds.sort_values(['id', 'Time [s]']).reset_index(drop=True)
    box_mae = mean_absolute_error(preds.iloc[:, 2:-1], truth.iloc[:, 2:-1])
    
    return box_mae

        
class BoxValidationCallback(Callback):
    
    def __init__(self, metric, exps, num_timesteps, in_array, env_array, y_scaler, output_cols, out_val):
        
        super(BoxValidationCallback, self).__init__()
        
        self.metric = metric
        self.exps = exps
        self.num_timesteps = num_timesteps 
        self.in_array = in_array
        self.env_array = env_array 
        self.y_scaler = y_scaler 
        self.output_cols = output_cols 
        self.out_val = out_val

    def on_train_begin(self, logs={}):
        logs = logs or {}
        self.best = 1e10
        self.best_epoch = 0
        self.box_loss = []

    def on_epoch_end(self, epoch, logs={}):
        logs = logs or {}
        box_loss = box_validate(
            self.model, 
            self.exps, 
            self.num_timesteps, 
            self.in_array, 
            self.env_array, 
            self.y_scaler, 
            self.output_cols, 
            self.out_val
        )
        self.box_loss.append(box_loss)
        logs[self.metric] = box_loss
        current = logs.get(self.metric)
        
        if np.less(self.best, current):
            self.best = box_loss
            self.best_epoch = epoch
            

class CustomModel(DenseNeuralNetwork):
        
    def fit(self, x, y, x_val, y_val, callbacks = [], rate = 1, scaler_type = "MinMaxScaler"):
        
        inputs = x.shape[1]
        outputs = [i.shape[-1] for i in y]
        
        self.build_neural_network(inputs, outputs, rate=rate, scaler_type=scaler_type)
        
        history = self.model.fit(
            x, y,
            validation_data=(x_val, y_val),
            batch_size=self.batch_size,
            epochs=self.epochs, 
            callbacks=callbacks,
            verbose=self.verbose)
        
        return history
    
    
    def build_neural_network(self, inputs, outputs, rate = 1, scaler_type = "MinMaxScaler"):
        """
        Create Keras neural network model and compile it.

        Args:
            inputs (int): Number of input predictor variables
            outputs (int): Number of output predictor variables
        """
        nn_input = Input(shape=(inputs,), name="input")
        nn_model = nn_input
        if self.activation == 'leaky':
            self.activation = LeakyReLU()
        if self.activation == 'prelu':
            self.activation = PReLU()

        if self.kernel_reg == 'l1':
            self.kernel_reg = l1(self.l1_weight)
        elif self.kernel_reg == 'l2':
            self.kernel_reg = l2(self.l2_weight)
        elif self.kernel_reg == 'l1_l2':
            self.kernel_reg = l1_l2(self.l1_weight, self.l2_weight)
        else:
            self.kernel_reg = None

        # Override for using custom activiation layers (use linear as last, then apply the lambda layer)
        use_custom_activation = False
        if "lambda" in self.output_activation:
            rate = float(self.output_activation.split(":")[-1])
            self.output_activation = "linear"
            use_custom_activation = True
            if scaler_type == "MinMaxScaler":
                custom_activation = CustomActivationExp(rate = rate)
            else:
                custom_activation = CustomActivationTanh(rate = rate)
            
        for h in range(self.hidden_layers):
            nn_model = Dense(self.hidden_neurons, activation=self.activation,
                             kernel_regularizer=self.kernel_reg, name=f"dense_{h:02d}")(nn_model)
            if self.use_dropout:
                nn_model = Dropout(self.dropout_alpha, name=f"dropout_h_{h:02d}")(nn_model)
            if self.use_noise:
                nn_model = GaussianNoise(self.noise_sd, name=f"ganoise_h_{h:02d}")(nn_model)
        nn_model_out = {}
        for i in range(len(outputs)):
            nn_model_out[i] = Dense(outputs[i],
                             activation=self.output_activation, name=f"dense_out_{i:02d}")(nn_model)
            if use_custom_activation:
                nn_model_out[i] = Lambda(custom_activation.out)(nn_model_out[i])
         
        output_layers = [x for x in nn_model_out.values()]
        self.model = Model(nn_input, output_layers)
#         if self.optimizer == "adam":
#             self.optimizer_obj = Adam(lr=self.lr)#, beta_1=self.adam_beta_1, beta_2=self.adam_beta_2,
#                                       #epsilon=self.epsilon, decay=self.decay)
#         elif self.optimizer == "sgd":
#             self.optimizer_obj = SGD(lr=self.lr)#, momentum=self.sgd_momentum, decay=self.decay)

        if self.loss == ['Xsigmoid']:
            self.model.compile(optimizer=self.optimizer, loss=self.x_sigmoid, loss_weights=self.loss_weights)
        elif self.loss == ['Xtanh']:
            self.model.compile(optimizer=self.optimizer, loss=x_tanh, loss_weights=self.loss_weights)
        else:
            self.model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=self.loss_weights)
            

class CustomActivationExp:
    def __init__(self, rate = 1):
        self.rate = rate
    def out(self, x):
        return 1.0 * (1.0 - K.exp(-self.rate * K.abs(x)))
    
class CustomActivationTanh:
    def __init__(self, rate = 1):
        self.rate = rate
    def out(self, x):
        return 5.0 * K.tanh(self.rate * x)

def x_tanh(y_actual, y_pred):
    x = y_actual - y_pred
    custom_loss = K.mean(x * K.tanh(x))
    return custom_loss