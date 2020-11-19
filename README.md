# GECKO-A Emulation

The gecko-ml package provides a framework for building machine learning
emulators of the GECKO-A chemistry model.
## Contributors
Charlie Becker; David John Gagne; Alma Hodzic; Keely Lawrence; John Schreck; Siyuan Wang



## Introduction

Natural and anthropogenic sources emit a large number of volatile organic compounds (VOCs). These compounds greatly affect the “self-cleaning capacity” of the atmosphere. These compounds can also undergo complicated chemical and physical processes in the atmosphere, forming organic aerosols. Organic aerosols have significant direct (absorbing/scattering) and indirect (affect cloud formation/properties) radiation effect.

In order to understand the broad impacts of VOCs on air quality and climate, we need to understand their sources and fates in the atmosphere. Many of these compounds can be directly emitted, or be produced from other compounds; in the meantime, they can undergo a variety of chemical reactions in the atmosphere. The chemical mechanism of these VOCs in the atmosphere can be very highly complicated, especially involving the formation of organic aerosols.

Most air quality models or chemistry-climate models are equipped with chemical mechanisms consisting of hundreds-thousands of chemical reactions. It is often found that such simplified chemical mechanisms are incapable to describe the complicity in the atmosphere. A widely used “near-explicit” chemical mechanism (Master Chemical Mechanism) consists of tens of thousands of chemical reactions, which still cannot capture the behavior and characteristics of the formation of organic aerosols. Such “near-explicit” mechanism is too computationally demanding for most air quality models or chemistry-climate models already.

NCAR (USA) and Centre National de la Recherche Scientifique (CNRS, France) jointly developed a hyper-explicit chemical mechanism: Generator of Explicit Chemistry and Kinetics of Organics in the Atmosphere (GECKO-A). GECKO-A can generate chemical mechanisms based on how the molecules/bonds/atoms interact with each other. Chemical mechanisms generated by GECKO-A usually include millions to tens of millions of reactions. Due to the remarkable computational cost, no air quality models or chemistry-climate models can afford to run with GECKO-A in the foreseeable future. There is a growing interest in the community to implement such complicated mechanisms into air quality models or chemistry-climate models, to study the broader impacts on air quality, human health, and the climate system.

Machine‐learning and artificial intelligence have proven to be a valuable tool in atmospheric science. Recent years have seen quite a few inspiring applications in developing machine-learning emulators using explicit/process-level models and implementing the trained emulators into large-scale models. Such explicit/process-level models are otherwise too expensive for large-scale models.

The goal of this project is to train the machine-learning emulator using the “library” generated by the hyper-explicit chemical mechanism, GECKO-A. 

## Data
 **Data generation procedure**: The machine-learning training dataset in this project (or library) is generated by the hyper-explicit chemical mechanism, GECKO-A. This library consists of results from thousands of GECKO-A simulations under vastly different conditions. Environmental conditions remain static for each GECKO-A box model experiment. More information about GECKO-A can be found [here](https://www2.acom.ucar.edu/modeling/gecko). Each model run lasts 5 simulation days.
 
 #### Potential Input Variables
* Precursor<sub>(*t*)</sub> (micrograms per cubic meter): Time series of the VOC precursor. 
* Gas<sub>(*t*)</sub> (micrograms per cubic meter): Mass concentration of products in the gas-phase.
* Aerosol<sub>(*t*)</sub> (micrograms per cubic meter): Mass concentration of products in the aerosol-phase.
* Temperature (K): The temperature at which the GECKO-A experiments were conducted. Temperature will affect the reaction rates of many reactions. It also affects the partitioning of a given compound between the gas-phase and particle-phase. Temerature the only environmental input feature that varies over a constant diurnal range.
* Solar zenith angle (degree): The solar zenith angle at which the GECKO-A experiments were conducted. This will affect the photolysis reactions. 
* Pre-existing aerosols (micrograms per cubic meter): Depending on the environmental conditions (e.g., temperature) and the vapor pressure, a compound can deposit onto (pre-existing) aerosols or evaporate from the aerosols. 
* NOx (ppb): concentration of nitrogen oxides. These are important compounds in the atmosphere, affecting the chemical mechanisms of many volatile organic compounds.
* O3 (ppb): concentration of ozone. It’s another important compounds in the atmosphere, affecting the chemical mechanisms of many VOCs.
* OH (10^6 molecules per cubic centimeter): concentration of hydroxyl radicals (OH). It’s one of the most important oxidants in the atmosphere, largely driving the oxidation of many VOCs.

#### Output Variables
* Precursor<sub>(*t+1*)</sub> (micrograms per cubic meter): Time series of the VOC precursor. 
* Gas<sub>(*t+1*)</sub> (micrograms per cubic meter): Mass concentration of products in the gas-phase.
* Aerosol<sub>(*t+1*)</sub> (micrograms per cubic meter): Mass concentration of products in the aerosol-phase.

*Gas and Aerosol represent the total combined mass of the chemical species. In the GECKO-A library, each phase is binned and output by volitility: 14 bins of each. In the configuaration file, there is an option to aggregate this data to simplify trainging or use each bin as its own feature and output.*

#### Metadata

| Metadata | Units | Label | 
| ------------- | :----:|:----------- | 
| Number Experiments   | 2000     | id | 
| Total Timesteps per experiment  | 1440     | Time |
| Timestep Delta   | 300 seconds | - |


## Requirements / Setup
```bash
git clone https://github.com/NCAR/gecko-ml.git
cd gecko-ml
pip install .
```
## Running 

Training and running the model in a forward-walking scenario are split into seperate procedures. If you're on Casper, you can run the following submit scripts.

`sbatch applications/train_gecko_emulator.sh`

followed by:

`sbatch applications/run_gecko_emulator.sh`

Each will pipe output to their respective `train.txt` / `run.txt` files. Output generated by the files will be in the `applications/save_out` directory. YAML configuration files can be found in the `./config/` directory and need to be passed to each train/run script with the -c flag. For the `run_gecko_emulator.py` script, additional flags can be passed to set the number of dask workers (-w) and number of threads-per-worker (-t) to speed of validation runs of the emulator. However, the default values should be okay for most scenarios. Note that the number of workers must be in line with the number of nodes requested (2 by default).

## Explanation of Configuaration Arguments
- Keys:
 - **species**: Chemical species (current libraries of 'dodecane', 'apin_O3', or 'toluene') 
 - **dir_path**: Top level directory path of species data 
 - **output_path**: Path to output models/data/etc.
 - **summary_file**: Species-specific name of summary file (CSV)
 - **aggregate_bins**: Boolean to determine if Mass will be aggregated or left binned by volitility (True/False or 1/0)
 - **save_models**: Boolean to determine if all models are saved or not
 - **ensemble_members**: (int) Number of ensemble members to run (must be >= 1).
 - **seq_length**: (int) How long of a sequence to use to predict a single timestep ahead using an LSTM model (only used in 'multi-time step models)
 - **scaler_type**: (str) Scaler type. Currently supports "MinMaxScaler" and "StandardScaler"
 - **bin_prefix**: Prefix of Gas/Aerosol to use when aggregating_bins is true
 - **min_exp**: (int) min exp number to pull data from (of 2000 total exps)
 - **max_exp**: (int) max exp number to pull data from (of 2000 total exps)
 - **num_exps**: (int or "all") Number of experiments to sample from validation set for box emulation. Must be no more than 10% of the total range between 'min_exp' and 'max_exp'
 - **input_vars**: List of all input variables to model (including Time and Experiment number)
 - **output_vars**: List of all output variables to model (including Time and Experiment number)


- **model_configurations**: Nested dictionary of different model types
  - **single_ts_models**: Nested group of single time step models (only use a single timestep to make a prediction)
    - **dnn_1**: Name of first model (can name key whatever) 
      - **hidden_layers**: (int) number of hidden layers (how deep the network is)
      - **hidden_neurons**: (int) Number of neurons per hidden layer
      - **activation**: (str) Activation type. Supports all keras layers that are identified by a "string" and also "leaky" for leaky ReLU (recommended)
      - **output_layers**: (int) Number of output layers (1, 2, or 3). Each layer represents "precursor", "gas" and "aerosol" masses (regardless of aggregation)
      - **output_activation**: Activation function for output layers
      - **optimizer**: Keras optimizer (supports "adam" or "sgd")
      - **loss**: (int or [ints]) Loss functions for each respective output layer
      - **loss_weights**: (int or [ints]) Loss weights for each respective output layer
      - **lr**: Optimizer learning rate
      - **batch_size**: Batch size for learning
      - **use_noise**: (boolean) Add noise or not to training data
      - **noise_sd**: (float) Standard deviation of white noise to be added 
      - **epochs**: (int) Number of epochs to run model
      - **use_dropout**: (boolean) Add dropout layer 
      - **dropout_alpha**: (float) percentage (decimal) of neurons to randomly drop during training
      - **l2_weight**: (float) L2 regularization amount
      - **sgd_momentum**: (float) Momentum amount for SGD optimizer
      - **adam_beta_1**: (float) Beta_1 amount for Adam optimizer
      - **adam_beta_2**: (float) Beta_2 amount for Adam optimizer
      - **decay**: (float) Amount to decay the learning rate 
      - **verbose**:  Verbosity of model traning (0 = None, 1 = minimal, 2 = max)
      - **classifier**: (boolean) classifier or regression
    - **multi-ts-models**: Nested group of multi time step models (use more than one timestep to make a prediction)
      - ...
        - ...