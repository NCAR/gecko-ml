# sys.path.insert(0, '/glade/work/cbecker/gecko-ml/')
import warnings
from geckoml.data import partition_y_output, load_data, transform_data, inv_transform_preds
from geckoml.box import GeckoBoxEmulator
from geckoml.metrics import ensembled_metrics
import logging
try:
    from echo.src.base_objective import *
    from echo.src.pruners import KerasPruningCallback
except ModuleNotFoundError:
    raise OSError("aimlutils does not seem to be installed, or is not on your python path. Exiting.")
from geckoml.models import DenseNeuralNetwork
from tensorflow.python.framework.ops import disable_eager_execution


warnings.filterwarnings("ignore")
disable_eager_execution()
logger = logging.getLogger(__name__)


class Objective(BaseObjective):

    def __init__(self, config, metric="box_mae", device="cpu"):

        BaseObjective.__init__(self, config, metric, device)

    def train(self, trial, conf):

        species = conf['species']
        dir_path = conf['dir_path']
        aggregate_bins = conf['aggregate_bins']
        input_vars = conf['input_vars']
        output_vars = conf['output_vars']
        log_trans_cols = conf['log_trans_cols']
        tendency_cols = conf['tendency_cols']
        scaler_type = conf['scaler_type']
        exps = conf['box_val_exps']

        data = load_data(dir_path, aggregate_bins, species, input_vars, output_vars, log_trans_cols)
        transformed_data, x_scaler, y_scaler = transform_data(data, None, species, tendency_cols, log_trans_cols,
                                                              scaler_type, output_vars, train=True)

        y = partition_y_output(transformed_data['train_out'].values, conf["MLP"]['output_layers'],
                               aggregate_bins)
        mod = DenseNeuralNetwork(**conf["MLP"])
        mod.fit(transformed_data['train_in'], y)

        box_mod = GeckoBoxEmulator(neural_net_path=None,
                                   input_cols=input_vars,
                                   output_cols=output_vars,
                                   model_object=mod,
                                   hyper_opt=True)

        raw_box_preds = box_mod.run_box_simulation(raw_val_output=data['val_out'],
                                                     transformed_val_input=transformed_data['val_in'],
                                                     exps=exps)

        truth, box_preds = inv_transform_preds(raw_preds=raw_box_preds,
                                                truth=data["val_out"],
                                                y_scaler=y_scaler,
                                                log_trans_cols=log_trans_cols,
                                                tendency_cols=tendency_cols)

        metrics = ensembled_metrics(y_true=truth,
                                    y_pred=box_preds,
                                    member=0,
                                    output_vars=output_vars,
                                    stability_thresh=10)

        box_mae = {'mean_mae': metrics['mean_mae'].mean(),
                   'mean_mape': metrics['Mean % MAE'].mean(),
                   'gas_mape': metrics.loc[metrics['mass_phase'] == 'Gas [ug/m3]', 'Mean % MAE'],
                   'aero_mape': metrics.loc[metrics['mass_phase'] == 'Aerosol [ug_m3]', 'Mean % MAE']}

        return box_mae
