from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import join
from .data import inverse_log_transform

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

def mean_abs_error(y_true, y_pred):
    """ Calculate the Mean Absolute Error
    Args:
        y_true (np.array): True output data
        y_pred (np.array): Predicted output data
    """
    return mean_absolute_error(y_true, y_pred)

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
    fig.savefig(join(output_path, 'plots', f'{species}_{model_name}_mae_ts.png'))

def ensembled_metrics(y_true, y_pred, member):
    """ Call a variety of metrics to be calculated (Hellenger distance R2, and RMSE currently) on Box emulator results.
        If bins were not aggregated, all bins are summed before metrics are calculated.
    Args:
        y_true (np.array): True output data
        y_pred (np.array): Predicted output data
    Returns:
        metrics (pd.dataframe): results for 'Precursor', 'Gas', and 'Aerosol' for a variety of metrics 
    """

    df = pd.DataFrame(columns=['ensemble_member', 'mass_phase', 'mean_mse', 'mean_mae', 'mean_r2', 'mean_hd', 'sd_mse',
                               'sd_mae', 'sd_r2', 'sd_hd', 'n_val_exps'])

    for col in y_true.columns[1:-1]:
        
        l = []
        l.append(member)
        l.append(col)
        l.append(mean_squared_error(y_true[col], y_pred[col]))
        l.append(mean_absolute_error(y_true[col], y_pred[col]))
        l.append(r2_corr(y_true[col], y_pred[col]))
        l.append(hellinger_distance(y_true[col], y_pred[col]))

        temp_df = pd.DataFrame(data={'t': y_true[col].values, 'p': y_pred[col].values, 'id': y_true['id']})
        l.append(temp_df.groupby('id').apply(lambda x: mean_squared_error(x['t'], x['p'])).std())
        l.append(temp_df.groupby('id').apply(lambda x: mean_absolute_error(x['t'], x['p'])).std())
        l.append(temp_df.groupby('id').apply(lambda x: r2_corr(x['t'], x['p'])).std())
        l.append(temp_df.groupby('id').apply(lambda x: hellinger_distance(x['t'], x['p'])).std())
        l.append(temp_df['id'].nunique())

        df = df.append(pd.DataFrame([l], columns=df.columns))

    return df


def match_true_exps(truth, preds, num_timesteps, seq_length, aggregate_bins, bin_prefix):
    """ Retrieve true values that match the experiments used in previous box emulator run
    Args:
        truth (DataFrame): True output data
        preds (DataFrame): Predicted output data
        num_timesteps (int): number of timesteps used in emulator runs to pull equivalent number from true data
        seq_length (int): Sequence length used in RNN/LSTM model
        aggregate_bins (boolean): Whether or not to aggregate data (determines number of features)
        bin_prefix (list): List of strings to aggregate on if aggregate_bins is True
    """
    true_df = truth.copy()
    if not aggregate_bins:
        for prefix in bin_prefix:
            true_df[prefix] = true_df.loc[:, true_df.columns.str.contains(prefix, regex=False)].sum(axis=1)
            preds[prefix] = preds.loc[:, preds.columns.str.contains(prefix, regex=False)].sum(axis=1)
        true_df = true_df[['Time [s]', 'Precursor [ug/m3]', 'Gas [ug/m3]', 'Aerosol [ug_m3]', 'id']]
        preds = preds[['Time [s]', 'Precursor [ug/m3]', 'Gas [ug/m3]', 'Aerosol [ug_m3]', 'id']]

    exps = preds['id'].unique()
    true_sub = true_df.loc[true_df['id'].isin(exps)]
    true_sub = true_sub.groupby('id').apply(lambda x: x.iloc[seq_length - 1: num_timesteps, :]).reset_index(drop=True)
    true_sub = true_sub.sort_values(['id', 'Time [s]']).reset_index(drop=True)
    preds = preds.sort_values(['id', 'Time [s]']).reset_index(drop=True)
    return true_sub, preds


def plot_ensemble(truth, preds, output_path, species, model_name):
    """ Plot ensemble members, ensemble mean, and truth from 3 randomly selected experiments.
    Args:
        truth: Validation dataframe for selected experiments
        preds: Validation dataframe of emulated results for selected experiments
        output_path: Path to save output
        species: Species (from config) used for labeling
        model_name: Model Name (used for labeling)
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
                p = preds[key][preds[key]['id'] == exps[j]].iloc[:, i + 1].values
                if dummy_i == 0:
                    axes[i, j].plot(p, linewidth=0.3, color=color[j], label='Ensemble Member')
                else:
                    axes[i, j].plot(p, linewidth=0.3, color=color[j], label='')
                dummy_i += 1
            m = mean_ensemble[mean_ensemble['id'] == exps[j]].iloc[:, i + 1].values
            axes[i, j].plot(m, color=color[j], linewidth=2, label='Ensemble Mean')
    for i in range(3):
        axes[0, i].legend()

    plt.savefig(join(output_path, 'plots', f'{species}_{model_name}_ensemble.png'), bbox_inches='tight')
