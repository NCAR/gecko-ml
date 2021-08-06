Files included:

1. gecko_shap.py
2. call.sh
3. summary.ipynb

1. 
The script gecko_shap.py will compute Shapley values for the Gecko models.

python gecko_shap.py --help

will print the parser options plus the default values. You need to supply:

model_config: the model configuration used to train a model
-s: a location path where the SHAP results will be stored
-m: a path to a pre-trained model 

Additional options allow you to spread the load across multiple nodes on 
casper or cheyenne. They are:

--workers: The number of nodes you want to use
--worker: The id of the current worker

If you plan to use 10 nodes, the worker option would range from 0 to 9
for the 10 jobs that get submitted. 

2. 
The script call.sh is an example demonstrating usage with the MLP and GRU models.

3. 
The notebook summary.ipynb will create a figure showing the bulk SHAP
values for a molecule, and a figure comparing three experiments.

If you use more than 1 worker, wait until all are finished before
computing summary figures with the notebook.