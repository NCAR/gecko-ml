from tensorflow.keras.layers import Input, Dense, Dropout, GaussianNoise, Activation, \
    Concatenate, BatchNormalization, LSTM, Conv1D, AveragePooling1D, MaxPooling1D, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.backend as K
from keras_self_attention import SeqSelfAttention
import tensorflow as tf
import numpy as np
import xarray as xr
import pandas as pd


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
                 l2_weight=0.01, sgd_momentum=0.9, adam_beta_1=0.9, adam_beta_2=0.999, decay=0, verbose=0,
                 classifier=False):
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

        for h in range(self.hidden_layers):
            nn_model = Dense(self.hidden_neurons, activation=self.activation,
                             kernel_regularizer=l2(self.l2_weight), name=f"dense_{h:02d}")(nn_model)
            if self.use_dropout:
                nn_model = Dropout(self.dropout_alpha, name=f"dropout_h_{h:02d}")(nn_model)
            if self.use_noise:
                nn_model = GaussianNoise(self.noise_sd, name=f"ganoise_h_{h:02d}")(nn_model)
        nn_model_out = {}
        for i in range(len(outputs)):
            nn_model_out[i] = Dense(outputs[i],
                             activation=self.output_activation, name=f"dense_out{i:02d}")(nn_model)
        output_layers = [x for x in nn_model_out.values()]
        self.model = Model(nn_input, output_layers)
        if self.optimizer == "adam":
            self.optimizer_obj = Adam(lr=self.lr, beta_1=self.adam_beta_1, beta_2=self.adam_beta_2, decay=self.decay)
        elif self.optimizer == "sgd":
            self.optimizer_obj = SGD(lr=self.lr, momentum=self.sgd_momentum, decay=self.decay)
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
        nn_ds.to_netcdf(filename, encoding={'layer_names': {'dtype': 'S1'}})
        return

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
                 classifier=False):
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
        seed = 8886
        #np.random.seed(seed)
        #tf.random.set_seed(seed)

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
                             activation=self.output_activation, name=f"lstm_out{i:02d}")(nn_model)
        output_layers = [x for x in nn_model_out.values()]
        self.model = Model(nn_input, output_layers)
        if self.optimizer == "adam":
            self.optimizer_obj = Adam(lr=self.lr, beta_1=self.adam_beta_1, beta_2=self.adam_beta_2, decay=self.decay)
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

