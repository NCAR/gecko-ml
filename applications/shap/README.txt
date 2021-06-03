Files included:

1. gecko_shap.py
2. summary.ipynb
3. call.sh
4. pbs_launch.sh
5. launch_workers.sh

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
The notebook summary.ipynb will create a figure showing the bulk SHAP
values for a molecule, and a figure comparing three experiments.

3. 
The script call.sh exemplifies how to call gecko_shap.py for different 
molecules and models.

### Multi-node options

4. 
The script pbs_launch.sh shows a submission script using 10 nodes. 
Edit the relevant fields in the script to toggle the number of workers,
and set the parser options for gecko_shap.py.

5. 
The script launch_workers.sh will submit N workers, where the number of workers
should be set to the number specified in pbs_launch.sh. launch_workers will create 
copies of pbs_launch.sh, update the worker field, and submit the job. 

For example, if you set workers=10 in pbs_launch.sh, in launch_workers.sh ensure
the for loop runs over {0..9} (e.g. workers -1).