from tensorflow.keras.layers import (Input, Dense, Dropout, GaussianNoise, 
                                     Activation, Concatenate, BatchNormalization, 
                                     LSTM, Conv1D, AveragePooling1D, MaxPooling1D, 
                                     LeakyReLU)

from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import Adam, SGD
from keras_self_attention import SeqSelfAttention
from tensorflow.keras.models import Model

import tensorflow.keras.backend as K
import tensorflow as tf
import xarray as xr
import pandas as pd
import numpy as np
import logging
import torch
import os


logger = logging.getLogger(__name__)


class DenseNeuralNetwork(object):
    """
    A Dense Neural Network Model that can support arbitrary numbers of hidden layers.

    Attributes:
        hidden_layers: Number of hidden layers
        hidden_neurons: Number of neurons in each hidden layer
        activation: Type of activation function
        output_layers: Number of output layers (1, 2, or 3)
        output_activation: Activation function applied to the output layer
        optimizer: Name of optimizer or optimizer object.
        loss: Name of loss functions or loss objects (can match up to number of output layers)
        loss_weights: Weights to be assigned to respective loss/output layer
        use_noise: Whether or not additive Gaussian noise layers are included in the network
        noise_sd: The standard deviation of the Gaussian noise layers
        lr: Learning rate for optimizer
        use_dropout: Whether or not Dropout layers are added to the network
        dropout_alpha: proportion of neurons randomly set to 0.
        batch_size: Number of examples per batch
        epochs: Number of epochs to train
        l2_weight: L2 weight parameter
        sgd_momentum: SGD optimizer momentum parameter
        adam_beta_1: Adam optimizer beta_1 parameter
        adam_beta_2: Adam optimizer beta_2 parameter
        decay: Level of decay to apply to learning rate
        verbose: Level of detail to provide during training (0 = None, 1 = Minimal, 2 = All)
        classifier: (boolean) If training on classes
    """
    def __init__(self, hidden_layers=1, hidden_neurons=4, activation="relu", output_layers=1,
                 output_activation="linear", optimizer="adam", loss="mse", loss_weights=1, use_noise=False,
                 noise_sd=0.01, lr=0.001, use_dropout=False, dropout_alpha=0.1, batch_size=128, epochs=2,
                 kernel_reg='l2', l1_weight=0.01, l2_weight=0.01, sgd_momentum=0.9, adam_beta_1=0.9, adam_beta_2=0.999,
                 epsilon=1e-7, decay=0, verbose=0, classifier=False):
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.output_layers = output_layers
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.optimizer_obj = None
        self.sgd_momentum = sgd_momentum
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.epsilon = epsilon
        self.loss = loss
        self.loss_weights = loss_weights
        self.lr = lr
        self.kernel_reg = kernel_reg
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.batch_size = batch_size
        self.use_noise = use_noise
        self.noise_sd = noise_sd
        self.use_dropout = use_dropout
        self.dropout_alpha = dropout_alpha
        self.epochs = epochs
        self.decay = decay
        self.verbose = verbose
        self.classifier = classifier
        self.y_labels = None
        self.model = None

    def x_sigmoid(self, y_actual, y_pred):
        x = y_actual - y_pred
        custom_loss = K.mean(2 * x / (1 + K.exp(-x)) - x)
        return custom_loss

    def build_neural_network(self, inputs, outputs):
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

        if self.kernel_reg == 'l1':
            self.kernel_reg = l1(self.l1_weight)
        elif self.kernel_reg == 'l2':
            self.kernel_reg = l2(self.l2_weight)
        elif self.kernel_reg == 'l1_l2':
            self.kernel_reg = l1_l2(self.l1_weight, self.l2_weight)
        else:
            self.kernel_reg = None

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
        output_layers = [x for x in nn_model_out.values()]
        self.model = Model(nn_input, output_layers)
        if self.optimizer == "adam":
            self.optimizer_obj = Adam(lr=self.lr, beta_1=self.adam_beta_1, beta_2=self.adam_beta_2,
                                      epsilon=self.epsilon, decay=self.decay)
        elif self.optimizer == "sgd":
            self.optimizer_obj = SGD(lr=self.lr, momentum=self.sgd_momentum, decay=self.decay)

        if self.loss == 'Xsigmoid':
            self.model.compile(optimizer=self.optimizer_obj, loss=self.x_sigmoid, loss_weights=self.loss_weights)
        else:
            self.model.compile(optimizer=self.optimizer_obj, loss=self.loss, loss_weights=self.loss_weights)


    def fit(self, x, y):
        inputs = x.shape[1]
        outputs = [i.shape[-1] for i in y]
        if self.classifier:
            outputs = np.unique(y).size
        self.build_neural_network(inputs, outputs)
        if self.classifier:
            self.y_labels = np.unique(y)
            y_class = np.zeros((y.shape[0], self.y_labels.size), dtype=np.int32)
            for l, label in enumerate(self.y_labels):
                y_class[y == label, l] = 1
            self.model.fit(x, y_class, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose)
        else:
            self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose, shuffle=True)
            print(self.model.summary())
        return

    def save_fortran_model(self, filename):

        nn_ds = xr.Dataset()
        num_dense, num_dense_out = 0, 0
        layer_names, out_bias, out_weights = [], [], []

        for layer in self.model.layers:

            if "dense" in layer.name and "out" not in layer.name:
                layer_names.append(layer.name)
                dense_weights = layer.get_weights()
                nn_ds[layer.name + "_weights"] = ((layer.name + "_in", layer.name + "_out"), dense_weights[0])
                nn_ds[layer.name + "_bias"] = ((layer.name + "_out",), dense_weights[1])
                nn_ds[layer.name + "_weights"].attrs["name"] = layer.name
                nn_ds[layer.name + "_weights"].attrs["activation"] = str(layer.get_config()["activation"])
                num_dense += 1

            elif "dense" in layer.name and "out" in layer.name:
                dense_weights = layer.get_weights()
                out_weights.append(dense_weights[0])
                out_bias.append(dense_weights[1])
                num_dense_out += 1

        layer_names.append(self.model.layers[-num_dense_out].name)
        concatenated_weights = np.concatenate(out_weights, axis=1)
        concatenated_bias = np.concatenate(out_bias)
        nn_ds[self.model.layers[-num_dense_out].name + "_weights"] = (
            (self.model.layers[-num_dense_out].name + "_in", self.model.layers[-num_dense_out].name + "_out"),
            concatenated_weights)
        nn_ds[self.model.layers[-num_dense_out].name + "_bias"] = (
            (self.model.layers[-num_dense_out].name + "_out",), concatenated_bias)
        nn_ds[self.model.layers[-num_dense_out].name + "_weights"].attrs["name"] = \
            self.model.layers[-num_dense_out].name
        nn_ds[self.model.layers[-num_dense_out].name + "_weights"].attrs["activation"] = \
            self.model.layers[-num_dense_out].get_config()["activation"]
        num_dense += 1
        nn_ds["layer_names"] = (("num_layers",), np.array(layer_names))
        nn_ds.attrs["num_layers"] = num_dense
        nn_ds.to_netcdf(filename, encoding={'layer_names': {'dtype': 'S1'}})
        return nn_ds

    def predict(self, x):
        if self.classifier:
            y_prob = self.model.predict(x, batch_size=self.batch_size)
            y_out = self.y_labels[np.argmax(y_prob, axis=1)]
        else:
            y_out = np.block(self.model.predict(x, batch_size=self.batch_size))
        return y_out

    def predict_proba(self, x):
        y_prob = self.model.predict(x, batch_size=self.batch_size)
        return y_prob


class LongShortTermMemoryNetwork(object):
    """
    A Long Short-Term Memory Neural Network Model that can support arbitrary numbers of hidden layers.

    Attributes:
        hidden_layers: Number of hidden layers
        hidden_neurons: Number of neurons in each hidden layer
        activation: Type of activation function
        output_layers: Number of output layers (1, 2, or 3)
        output_activation: Activation function applied to the output layer
        optimizer: Name of optimizer or optimizer object.
        loss: Name of loss functions or loss objects (can match up to number of output layers)
        loss_weights: Weights to be assigned to respective loss/output layer
        use_noise: Whether or not additive Gaussian noise layers are included in the network
        noise_sd: The standard deviation of the Gaussian noise layers
        lr: Learning rate for optimizer
        use_dropout: Whether or not Dropout layers are added to the network
        dropout_alpha: proportion of neurons randomly set to 0.
        batch_size: Number of examples per batch
        epochs: Number of epochs to train
        l2_weight: L2 weight parameter
        sgd_momentum: SGD optimizer momentum parameter
        adam_beta_1: Adam optimizer beta_1 parameter
        adam_beta_2: Adam optimizer beta_2 parameter
        decay: Level of decay to apply to learning rate
        verbose: Level of detail to provide during training (0 = None, 1 = Minimal, 2 = All)
    """
    def __init__(self, hidden_layers=1, hidden_neurons=50, activation="relu", output_layers=1,
                 output_activation="linear", optimizer="adam", loss="mse", loss_weights=1,
                 use_noise=False, noise_sd=0.01, lr=0.001, use_dropout=False, dropout_alpha=0.1, batch_size=128,
                 epochs=2, l2_weight=0.01, sgd_momentum=0.9, adam_beta_1=0.9, adam_beta_2=0.999, decay=0, verbose=0,
                 epsilon=1e-7, classifier=False):
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.output_layers = output_layers
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.optimizer_obj = None
        self.sgd_momentum = sgd_momentum
        self.adam_beta_1 = adam_beta_1
        self.adam_beta_2 = adam_beta_2
        self.epsilon = epsilon
        self.loss = loss
        self.loss_weights = loss_weights
        self.lr = lr
        self.l2_weight = l2_weight
        self.batch_size = batch_size
        self.use_noise = use_noise
        self.noise_sd = noise_sd
        self.use_dropout = use_dropout
        self.dropout_alpha = dropout_alpha
        self.epochs = epochs
        self.decay = decay
        self.verbose = verbose
        self.classifier = classifier
        self.y_labels = None
        self.model = None

    def build_neural_network(self, seq_input, inputs, outputs):
        """
        Create Keras neural network model and compile it.

        Args:
            inputs (int): Number of input predictor variables
            outputs (int): Number of output predictor variables
            seq_input (int): Number of timesteps (length of sequence)
        """

        nn_input = Input(shape=(seq_input, inputs), name="input")
        nn_model = nn_input
        nn_model = Conv1D(64, 2, strides=1, padding='valid')(nn_model)
        for h in np.arange(self.hidden_layers):
            if h == np.arange(self.hidden_layers)[-1]:
                nn_model = LSTM(self.hidden_neurons, return_sequences=True, dropout=self.dropout_alpha,
                                name=f"lstm_{h:02d}")(nn_model)
                nn_model = SeqSelfAttention()(nn_model)
            else:
                nn_model = LSTM(self.hidden_neurons, return_sequences=True, dropout=self.dropout_alpha,
                                name=f"lstm_{h:02d}")(nn_model)
                nn_model = SeqSelfAttention()(nn_model)
            if self.use_noise:
                nn_model = GaussianNoise(self.noise_sd, name=f"ganoise_h_{h:02d}")(nn_model)
        nn_model_out = {}
        for i in range(len(outputs)):
            nn_model_out[i] = LSTM(outputs[i],
                             activation=self.output_activation, name=f"lstm_out_{i:02d}")(nn_model)
        output_layers = [x for x in nn_model_out.values()]
        self.model = Model(nn_input, output_layers)
        if self.optimizer == "adam":
            self.optimizer_obj = Adam(lr=self.lr, beta_1=self.adam_beta_1, beta_2=self.adam_beta_2,
                                      epsilon=self.epsilon, decay=self.decay)
        elif self.optimizer == "sgd":
            self.optimizer_obj = SGD(lr=self.lr, momentum=self.sgd_momentum, decay=self.decay)
        self.model.compile(optimizer=self.optimizer_obj, loss=self.loss, loss_weights=self.loss_weights)

    def fit(self, x, y):

        lookback = x.shape[1]
        n_features = x.shape[2]
        outputs = [i.shape[-1] for i in y]
        if self.classifier:
            outputs = np.unique(y).size
        self.build_neural_network(lookback, n_features, outputs)
        if self.classifier:
            self.y_labels = np.unique(y)
            y_class = np.zeros((y.shape[0], self.y_labels.size), dtype=np.int32)
            for l, label in enumerate(self.y_labels):
                y_class[y == label, l] = 1
            self.model.fit(x, y_class, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose)
        else:
            self.model.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose, shuffle=True)
            print(self.model.summary())
        return

    def save_fortran_model(self, filename):
        nn_ds = xr.Dataset()
        num_dense = 0
        layer_names = []
        for layer in self.model.layers:
            if "dense" in layer.name:
                layer_names.append(layer.name)
                dense_weights = layer.get_weights()
                nn_ds[layer.name + "_weights"] = ((layer.name + "_in", layer.name + "_out"), dense_weights[0])
                nn_ds[layer.name + "_bias"] = ((layer.name + "_out",), dense_weights[1])
                nn_ds[layer.name + "_weights"].attrs["name"] = layer.name
                nn_ds[layer.name + "_weights"].attrs["activation"] = layer.get_config()["activation"]
                num_dense += 1
        nn_ds["layer_names"] = (("num_layers",), np.array(layer_names))
        nn_ds.attrs["num_layers"] = num_dense
        nn_ds.to_netcdf(filename, encoding={'layer_names':{'dtype': 'S1'}})
        return


    def predict(self, x):
        if self.classifier:
            y_prob = self.model.predict(x, batch_size=self.batch_size)
            y_out = self.y_labels[np.argmax(y_prob, axis=1)].ravel()
        else:
            y_out = np.block(self.model.predict(x, batch_size=self.batch_size))
        return y_out



class GRUNet(torch.nn.Module):
    
    """
    A GRU Neural Network Model that can support arbitrary numbers of hidden layers.

    Attributes:
        hidden_neurons: int
        - Number of neurons in each hidden layer
        hidden_layers: int
        - Number of hidden layers
        drop_prob: float
        - proportion of neurons randomly set to 0.
        device: str
        - CPU or GPU identifier
        
        gru: torch.nn.Module
        - Torch GRU model
        fc: torch.nn.Module
        - Torch Linear layer for resizing the output of the GRU
        relu: torch.nn.Module
        - The activation on the GRU output 
        hidden_model: torch.nn.Module
        - Torch Linear layer for the initial hidden state
    """
    
    def __init__(self, 
                 hidden_dim: int,
                 n_layers: int, 
                 drop_prob: float = 0.2): 
        
        super(GRUNet, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dr = drop_prob
        self.device = None
        
        self.gru = None
        self.fc = None
        self.relu = None
        self.hidden_model = None
        
    def build(self, 
              input_dim: int, 
              output_dim: int,
              weights_path: str = None) -> None:
        
        """
            Build the GRU network 
        
            input_dim: int
            - The size of the input
            output_dim: int
            - The number of prediction targets
        """
        
        self.gru = torch.nn.GRU(input_dim, self.hidden_dim, self.n_layers, batch_first=True, dropout=self.dr)
        self.fc = torch.nn.Linear(self.hidden_dim, output_dim)
        self.relu = torch.nn.LeakyReLU()
        self.hidden_model = torch.nn.Linear(input_dim, self.hidden_dim)
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"The model contains {total_params} total parameters, {trainable_params} are trainable"
        )
        
        if isinstance(weights_path, str):
            if os.path.isfile(weights_path):
                self.load_weights(weights_path)
            else:
                logger.info(f"Failed to load model weights at {weights_path}")
                
        
    def load_weights(self, weights_path: str) -> None:
        
        """
            Loads model weights given a valid weights path
            
            weights_path: str
            - File path to the weights file (.pt)
        """
        logger.info(f"Loading model weights from {weights_path}")
        
        try:
            checkpoint = torch.load(
                weights_path,
                map_location=lambda storage, loc: storage
            )
            self.load_state_dict(checkpoint["model_state_dict"])
        except Exception as E:
            logger.info(
                f"Failed to load model weights at {weights_path} due to error {str(E)}"
            )
        
        
    def forward(self, 
                x: torch.Tensor, 
                h: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        
        """
            Pass the inputs through the model and return the prediction
        
            Inputs
            x: torch.Tensor
            - The input containing precursor, gas, aerosol, and envionmental values
            h: torch.Tensor
            - The hidden state to the GRU at time t
            
            Returns
            out: torch.Tensor
            - The encoded input
            h: torch.Tensor
            - The hidden state returned by the GRU at time t + 1
        """
        
        x = x.unsqueeze(1)
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:,-1]))
        return out, h
    
    def init_hidden(self, 
                    x: torch.Tensor) -> torch.Tensor:
        
        """
            Predict a hidden state for the initial input to the model at t = 0
        
            Inputs
            x: torch.Tensor
            - The input containing precursor, gas, aerosol, and envionmental values
            
            Returns: torch.Tensor
            - A hidden state corresponding to the initial condition
        """
        
        device = self._device() if self.device is None else self.device
        hidden = self.hidden_model(x.to(device)).unsqueeze(0)
        hidden = torch.cat([hidden for x in range(self.n_layers)]) if self.n_layers > 1 else hidden
        return hidden
    
    def _device(self) -> str:
        
        """
            Set and return the device that the model was placed onto.
        
            Inputs: None
            Returns: str
            - Device identifier
        
        """
        
        self.device = next(self.parameters()).device
        return self.device
    
    def predict(self, 
                x: np.array, 
                h: np.array) -> (np.array, np.array):
        
        """
            Predict method for running the model in box mode.
            Handles converting numpy tensor input to torch
            and moving the data to the GPU
        
            Inputs
            x: np.array
            - The input containing precursor, gas, aerosol, and envionmental values
            h: np.array
            - The hidden state to the GRU at time t
            
            Returns: str
            x: np.array
            - The encoded input
            h: np.array
            - The hidden state to the GRU at time t + 1
        
        """
        
        device = self._device() if self.device is None else self.device
        with torch.no_grad():
            x = torch.from_numpy(x).float().to(device)
            x, h = self.forward(x, h)
        return x.cpu().detach().numpy(), h