from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def calc_pdf_hist(x, x_bins):
    """ Calculate Probability Density Function. Normalized as the integral over the range == 1
    Args:
        x (np.array): Data to calculate PDF over.
        x_bins (np.array): Array of bins to calculate PDF.
    """
    return np.histogram(x, x_bins, density=True)[0]


def hellinger(x, pdf_p, pdf_q):
    """ Calculate Hellenger integral
    Args:
        x (np.array): Bin centers of all data
        pdf_p (np.array): Probability density function of true values
        pdf_q (np.array): Probability density funciton of predictions
    """
    pdf_distances = (np.sqrt(pdf_p) - np.sqrt(pdf_q)) ** 2
    return np.trapz(pdf_distances, x) / 2


def root_mean_squared_error(y_true, y_pred):
    """ Calculate the Root Mean Squared Error
    Args:
        y_true (np.array): True output data
        y_pred (np.array): Predicted output data
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def hellinger_distance(y_true, y_pred, bins=50):
    """ Calculate Hellenger distance between two distributions
    Args:
        y_true (np.array): True output data
        y_pred (np.array): Predicted output data
        bins (int): number of bins to calculate HD over
    """
    bin_points = np.linspace(np.minimum(y_true.min(), y_pred.min()),
                             np.maximum(y_true.max(), y_pred.max()), bins)
    bin_centers = 0.5 * (bin_points[:-1] + bin_points[1:])
    y_true_pdf = calc_pdf_hist(y_true, bin_points)
    y_pred_pdf = calc_pdf_hist(y_pred, bin_points)
    return hellinger(bin_centers, y_true_pdf, y_pred_pdf)


def r2_corr(y_true, y_pred):
    """ Calculate the coefficient of determination (R-squared).
    Args:
        y_true (np.array): True output data
        y_pred (np.array): Predicted output data
    """
    return np.corrcoef(y_true, y_pred)[0, 1] ** 2


def mae_time_series(y_true, y_pred):
    """ Calculate the Mean Absolute Error across timesteps
    Args:
        y_true (np.array): True output data
        y_pred (np.array): Predicted output data
    """
    time_series = y_true['Time [s]']
    y_true = y_true.sort_values(['id', 'Time [s]'], ascending=True).iloc[:, 1:-1]
    y_pred = y_pred.sort_values(['id', 'Time [s]'], ascending=True).iloc[:, :-2]

    if len(y_true.columns) > 5:
        preds = np.empty((y_pred.shape[0], 3))
        preds[:, 0] = y_pred.iloc[:, 0]
        preds[:, 1] = np.sum(y_pred.iloc[:, 0:14], axis=1)
        preds[:, 2] = np.sum(y_pred.iloc[:, 14:-2], axis=1)

        truth = np.empty((y_true.shape[0], 3))
        truth[:, 0] = y_true.iloc[:, 1]
        truth[:, 1] = np.sum(y_true.iloc[:, 2:16], axis=1)
        truth[:, 2] = np.sum(y_true.iloc[:, 16:-1], axis=1)

        cols = ['Precursor [ug/m3]', 'Gas [ug/m3]', 'Aerosol [ug_m3]']
        y_true, y_pred = pd.DataFrame(truth, columns=cols), pd.DataFrame(preds, columns=cols)

    df_diff = np.abs(y_true - y_pred)

    df_diff['Time [s]'] = time_series
    mae = df_diff.groupby('Time [s]').mean()

    return mae

def plot_mae_ts(y_true, y_pred, output_path, model_name, species):
    """ Plot and save an average mean absolute error per timestep, across experiments
    Args:
        y_true (np.array): True output data
        y_pred (np.array): Predicted output data
        output_path (str): Output path top save figure to
        model_name (str): Model name used to label figure
    """

    mae = mae_time_series(y_true, y_pred)
    ax = mae.plot()
    ax.set_title('{} - MAE per Timestep'.format(model_name))
    fig = ax.get_figure()
    fig.savefig('{}plots/{}_{}_mae_ts.png'.format(output_path, species, model_name), bbox_inches='tight')

def ensembled_box_metrics(y_true, y_pred):
    """ Call a variety of metrics to be calculated (Hellenger distance R2, and RMSE currently) on Box emulator results.
        If bins were not aggregated, all bins are summed before metrics are calculated.
    Args:
        y_true (np.array): True output data
        y_pred (np.array): Predicted output data
    Returns:
        metrics (dictionary): results for 'Precursor', 'Gas', and 'Aerosol' for each type of metric.
    """
    if len(y_true.columns) > 5:
        preds = np.empty((y_pred.shape[0], 3))
        preds[:, 0] = y_pred.iloc[:, 0]
        preds[:, 1] = np.sum(y_pred.iloc[:, 0:14], axis=1)
        preds[:, 2] = np.sum(y_pred.iloc[:, 14:-2], axis=1)

        truth = np.empty((y_true.shape[0], 3))
        truth[:, 0] = y_true.iloc[:, 1]
        truth[:, 1] = np.sum(y_true.iloc[:, 2:16], axis=1)
        truth[:, 2] = np.sum(y_true.iloc[:, 16:-1], axis=1)

    else:
        preds = y_pred.sort_values(['id', 'Time [s]'], ascending=True).iloc[:, :-2].values
        truth = y_true.sort_values(['id', 'Time [s]'], ascending=True).iloc[:, 1:-1].values

    metrics = {}
    rmse_l, r2_l, hd_l = [], [], []

    for i in np.arange(3):
        rmse_l.append(root_mean_squared_error(truth[:, i], preds[:, i]))
    metrics['RMSE'] = rmse_l
    for i in np.arange(3):
        r2_l.append(r2_corr(truth[:, i], preds[:, i]))
    metrics['R2'] = r2_l
    for i in np.arange(3):
        hd_l.append(hellinger_distance(truth[:, i], preds[:, i]))
    metrics['HD'] = hd_l

    return metrics


def ensembled_base_metrics(y_true, y_pred, ids, seq_length=1):
    """ Call a variety of metrics to be calculated (Hellenger distance and RMSE, currently) on Base Model results using
        all output variables.
    Args:
        y_true (np.array): True output data
        y_pred (np.array): Predicted output data
    """
    y_pred['id'] = ids
    if seq_length > 1:
        y_true = y_true.groupby('id').apply(lambda x: x.iloc[seq_length:, 1:-1]).values

    else:
        y_true = y_true.iloc[:, 1:-1].values

    y_pred = y_pred.iloc[:, :-1].values
    metrics = {}

    metrics['RMSE'] = root_mean_squared_error(y_true, y_pred)
    #metrics['R2'] = root_mean_squared_error(y_true, y_pred)
    metrics['HD'] = hellinger_distance(y_true, y_pred)

    return metrics

def match_true_exps(truth, preds, num_timesteps, seq_length):
    """ Retrieve true values that match the experiments used in previous box emulator run
    Args:
        truth (DataFrame): True output data
        preds (DataFrame): Predicted output data
        num_timesteps (int): number of timesteps used in emulator runs to pull equivalent number from true data
        seq_length (int): Sequence length used in RNN/LSTM model
    """
    exps = preds['id'].unique()
    true_sub = truth.loc[truth['id'].isin(exps)]
    true_sub = true_sub.groupby('id').apply(lambda x: x.iloc[seq_length - 1: num_timesteps, :]).reset_index(drop=True)
    true_sub = true_sub.sort_values(['id', 'Time [s]'])

    preds = preds.sort_values(['id', 'Time [s]']).reset_index(drop=True)
    columns_dict = {x: y for x,y in zip(preds.columns[:-2], truth.columns[1:-1])}
    preds.rename(columns=columns_dict, inplace=True)

    return true_sub, preds


def plot_ensemble(truth, preds, output_path, species, model_name):
    """ Plot ensemble members, ensemble mean, and truth from 3 randomly selected experiments.
    """
    all_exps = truth['id'].unique()
    exps = np.random.choice(all_exps, 3, replace=False)
    color = ['r', 'b', 'g']
    mean_ensemble = pd.concat([x for x in preds.values()]).groupby(level=0).mean()
    mean_ensemble['id'] = truth['id']
    fig, axes = plt.subplots(3, 3, figsize=(20, 16), sharex='col', sharey='row',
                             gridspec_kw={'hspace': 0, 'wspace': 0})
    fig.suptitle('Ensemble Runs - {} - {}'.format(species, model_name), fontsize=30)

    for i in range(3):
        for j in range(3):
            t = truth[truth['id'] == exps[j]].iloc[:, i + 1].values
            axes[i, j].plot(t, linestyle='--', color='k', linewidth=2, label='True')
            if i == 0:
                axes[i, j].set_title(exps[j], fontsize=22)
            if j == 0:
                axes[i, j].set_ylabel(truth.columns[i+1], fontsize=20)
            dummy_i = 0
            for key, value in preds.items():
                p = preds[key][preds[key]['id'] == exps[j]].iloc[:, i].values
                if dummy_i == 0:
                    axes[i, j].plot(p, linewidth=0.3, color=color[j], label='Ensemble Member')
                else:
                    axes[i, j].plot(p, linewidth=0.3, color=color[j], label='')
                dummy_i += 1
            m = mean_ensemble[mean_ensemble['id'] == exps[j]].iloc[:, i].values
            axes[i, j].plot(m, color=color[j], linewidth=2, label='Ensemble Mean')
    for i in range(3):
        axes[0, i].legend()

    plt.savefig('{}plots/{}_{}_ensembled_exps.png'.format(output_path, species, model_name), bbox_inches='tight')
