from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import properscoring as ps
from os.path import join
from scipy.stats import pearsonr

from scipy.signal import tukey
from numpy.fft import fft, fftshift
from numpy.fft import rfft, rfftfreq


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


def get_outliers(preds, truth, cols, n_extremes=10):
    """
    Get best/worst experiments wrt to MAE from box simulations
    Args:
        preds: Box model predictions (df)
        truth: Box model truth (df)
        cols: Features to caluculate best/worst MAE
        n_extremes: Number of experiments to flag from each extreme

    Returns: best/worst experiments from box simulation (list)

    """
    def mae_wrap(df, cols):
        pred_cols = [x + '_pred' for x in cols]
        MAE = mean_absolute_error(df[cols], df[pred_cols])
        return MAE

    df_p = preds.copy()
    df_t = truth.copy()
    df_t = df_t.set_index(['id', 'member', 'Time [s]'])
    df_p = df_p.set_index(['id', 'member', 'Time [s]'])
    df = df_t.join(df_p, rsuffix='_pred')

    # check for inf values and collect exps with inf
    i = df.index[np.isinf(df[cols]).any(1)]
    inf_exps = list(np.unique(i.get_level_values(0)))

    if len(inf_exps) != 0:
        df = df.drop(index=inf_exps, level=0)

    errors = df.groupby('id').apply(mae_wrap, cols)
    best_exps = list(errors.sort_values()[:n_extremes].index)
    worst_exps = list(errors.sort_values()[-n_extremes:].index)
    worst_exps[:len(inf_exps)] = inf_exps

    return best_exps, worst_exps

def get_stability(preds, stability_thresh, output_cols):
    """
    Determine if any value has crossed the positive or negative magnitude of threshold and lable unstable if true
    Args:
        preds (pd.DataFrame): Predictions
        stability_thresh: Threshold to determine if an exp has gone unstable (uses positive and negative values)

    Returns:
        stable_exps (list)
        unstable_exps (list)
    """
    unstable = preds.groupby('id')[output_cols].apply(
        lambda x: x[(x > stability_thresh) | (x < -stability_thresh)].any())
    stable_exps = unstable[unstable == False].index
    unstable_exps = unstable[unstable == True].index

    return stable_exps, unstable_exps


def ensembled_metrics(y_true, y_pred, member, output_vars, stability_thresh=1):
    """ Call a variety of metrics to be calculated (Hellenger distance R2, and RMSE currently) on Box emulator results.
        If bins were not aggregated, all bins are summed before metrics are calculated.
    Args:
        y_true (np.array): True output data
        y_pred (np.array): Predicted output data
    Returns:
        metrics (pd.dataframe): results for 'Precursor', 'Gas', and 'Aerosol' for a variety of metrics 
    """
    y_pred_copy = y_pred.copy()
    for col in output_vars:
        y_pred_copy = y_pred_copy.groupby('id').filter(
            lambda x: (x[col].max() < stability_thresh) & (x[col].min() > -stability_thresh))

    stable_exps = y_pred_copy.index.unique(level='id')
    stable_true = y_true.loc[y_true.index.isin(stable_exps, level='id')]
    if len(stable_true.index.unique(level='id')) == 0:
        raise ValueError('No stable experiments to calculate metrics on! Aborting.')
    n_unstable = len(y_true.index.unique(level='id')) - len(stable_true.index.unique(level='id'))

    df = pd.DataFrame(columns=['ensemble_member', 'mass_phase', 'mean_mse', 'mean_mae', 'Mean % MAE', 'mean_r2',
                               'mean_pearson', 'mean_hd', 'n_val_exps', 'n_unstable'])

    for col in y_pred.columns:

        l = []
        l.append(member)
        l.append(col) 
        l.append(mean_squared_error(stable_true[col], y_pred_copy[col]))
        l.append(mean_absolute_error(stable_true[col], y_pred_copy[col]))
        l.append(mean_absolute_percentage_error(stable_true[col], y_pred_copy[col]))
        l.append(r2_corr(stable_true[col], y_pred_copy[col]))
        l.append(pearsonr(stable_true[col], y_pred_copy[col])[0])
        l.append(hellinger_distance(stable_true[col], y_pred_copy[col]))

        temp_df = pd.DataFrame(data={'t': stable_true[col].values, 'p': y_pred_copy[col].values,
                                     'id': stable_true.index.get_level_values(level='id')})
        l.append(temp_df['id'].nunique())
        l.append(n_unstable)

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

    preds['Time [s]'] = truth['Time [s]'].values
    exps = preds['id'].unique()
    true_sub = true_df.loc[true_df['id'].isin(exps)]
    true_sub = true_sub.groupby('id').apply(lambda x: x.iloc[seq_length - 1: num_timesteps, :]).reset_index(drop=True)
    true_sub = true_sub.sort_values(['id', 'Time [s]']).reset_index(drop=True)
    preds = preds.sort_values(['id', 'Time [s]']).reset_index(drop=True)
    return true_sub, preds


def plot_ensemble(truth, preds, output_path, output_vars, species, model_name):
    """ Plot ensemble members, ensemble mean, and truth from 3 randomly selected experiments.
    Args:
        truth: Validation dataframe for selected experiments
        preds: Validation dataframe of emulated results for selected experiments
        output_path: Path to save output
        species: Species (from config) used for labeling
        model_name: Model Name (used for labeling)
    """
    all_exps = truth.index.unique(level='id')
    exps = np.random.choice(all_exps, 3, replace=False)
    color = ['r', 'b', 'g']
    mean_ensemble = preds.groupby(['Time [s]', 'id']).mean()[output_vars]
    fig, axes = plt.subplots(len(output_vars), 3, figsize=(20, 16), sharex='col', sharey='row',
                             gridspec_kw={'hspace': 0, 'wspace': 0})
    fig.suptitle('Ensemble Runs - {} - {}'.format(species, model_name), fontsize=30)
    t_sub = truth[truth['member'] == 0]

    for i, exp in enumerate(exps):
        for j, var in enumerate(output_vars):
            t = t_sub.loc[t_sub.index.get_level_values('id') == exp, var].values
            axes[j, i].plot(t, linestyle='--', color='k', linewidth=2, label='True')
            if j == 0:
                axes[j, i].set_title(exp, fontsize=22)
            if i == 0:
                axes[j, i].set_ylabel(var, fontsize=20)
            for member in preds['member'].unique():
                p = preds.loc[preds['member'] == member]
                p_sub = p.loc[p.index.get_level_values('id') == exp, var].values
                if member == 0:
                    axes[j, i].plot(p_sub, linewidth=0.3, color=color[j], label='Ensemble Member')
                else:
                    axes[j, i].plot(p_sub, linewidth=0.3, color=color[j], label='')
            m = mean_ensemble.loc[mean_ensemble.index.get_level_values('id') == exp, var].values
            axes[j, i].plot(m, color=color[j], linewidth=2, label='Ensemble Mean')
    for i in range(3):
        axes[0, i].legend()

    plt.savefig(join(output_path, 'plots', f'{species}_{model_name}_ensemble.png'), bbox_inches='tight')


def bootstrap_ci(truth, preds, columns, n_bs_samples, ci_level):
    """
       Calculate MAE across validation experiments for single ensemble member
       Args:
           truth (pd.DataFrame): Modeled observations
           preds (pd.DataFrame): Predictions
           columns: Columns to plot MAE
           n_bs_samples (int): Number of bootstrap resamples
           ci_level (float): Confidence level in decimal form
       Returns:
            (Dicts): Mean MAE, Lower CI MAE, Upper CI MAE
       """
    err = (truth.loc[:, columns] - preds.loc[:, columns]).abs()
    err['Time [s]'] = truth['Time [s]'].values
    err['id'] = truth['id'].values
    mean_err = err.groupby('Time [s]').mean()
    err = err.set_index(['id', 'Time [s]']).unstack('id')

    min_quant = (1 - ci_level) / 2
    max_quant = 1 - min_quant

    n_exps = truth['id'].nunique()
    error_dict, lower_ci, upper_ci = {}, {}, {}

    for phase in columns:

        l = []

        for i in range(n_bs_samples):
            bs = err[phase].sample(n_exps, replace=True, axis=1).mean(axis=1)
            l.append(bs)

        all_bs = np.vstack(l)
        error_dict[phase] = mean_err[phase].values
        lower_ci[phase] = np.quantile(all_bs, min_quant, axis=0)
        upper_ci[phase] = np.quantile(all_bs, max_quant, axis=0)

    return error_dict, lower_ci, upper_ci


def plot_bootstrap_ci(truth, preds, columns, output_path, species, model_name, n_bs_samples=10000, ci_level=0.95,
                      only_stable=True, stable_thresh=1):
    """
       Plot MAE across validation experiments for single ensemble member
       Args:
           truth (pd.DataFrame): Modeled observations
           preds (pd.DataFrame): Predictions
           columns: Columns to plot MAE of
           output_path (str): Base path to save to
           species (str): Species name (used for titling and file naming)
           model_name (str): Model name (used for titling and file naming)
           n_bs_samples (int): Number of bootstrap resamples
           ci_level (float): Confidence level in decimal form
           only_stable (bool): If only stable experiments should be used
           stable_thresh (float): Magnitude to determine if experiments are stable (only used if only_stable=True)

       Returns:

       """
    if only_stable:
        stable_exps = get_stability(preds, stable_thresh, columns)[0]
        truth = truth[truth['id'].isin(stable_exps)]
        preds = preds[preds['id'].isin(stable_exps)]
        truth = truth[truth['member'] == 0]
        preds = preds[preds['member'] == 0]

    truth = truth[truth['member'] == 0]
    preds = preds[preds['member'] == 0]

    mean_err, lower_ci, upper_ci = bootstrap_ci(truth, preds, columns, n_bs_samples, ci_level)
    mean_truth = truth.groupby('Time [s]').mean()

    fig, axs = plt.subplots(len(columns), 1, figsize=(22, 16), sharex=True)
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.subplots_adjust(top=0.95)

    for i, ax in enumerate(axs.ravel()):

        colors = ['r', 'g', 'b']
        time = mean_truth.index / 60 / 60 / 24
        ax.xaxis.set_tick_params(labelsize=16)
        ax.plot(time, mean_err[columns[i]], color=colors[i], lw=3, label=columns[i])
        ax.plot(time, upper_ci[columns[i]], color='k', lw=0.5)
        ax.plot(time, lower_ci[columns[i]], color='k', lw=0.5)
        ax.fill_between(time, lower_ci[columns[i]], upper_ci[columns[i]], alpha=0.5, color='grey')
        ax.set_ylabel(columns[i], fontsize=16)
        ax.legend(loc=(0.7, 0.85), prop={'size': 16})

        if i == len(columns) - 1:
            ax.set_xlabel('Time [Days]', fontsize=20)
        fig.suptitle(f'Bootstapped {int(ci_level * 100)}% Confidence Intervals of MAE Estimates - {species}',
                     fontsize=24)
        plt.savefig(join(output_path, f'plots/{species}_MAE_{model_name}.png'), bbox_inches='tight')

    fig, axs = plt.subplots(len(columns), 1, figsize=(22, 16), sharex=True)
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.subplots_adjust(top=0.95)

    for i, ax in enumerate(axs.ravel()):

        colors = ['r', 'g', 'b']
        time = mean_truth.index / 60 / 60 / 24
        ax.xaxis.set_tick_params(labelsize=16)
        ax.plot(time, mean_truth[columns[i]], color=colors[i], lw=3, label=columns[i])
        ax.plot(time, mean_truth[columns[i]] + mean_err[columns[i]], color='k', lw=0.5)
        ax.plot(time, mean_truth[columns[i]] - mean_err[columns[i]], color='k', lw=0.5)
        ax.plot(time, mean_truth[columns[i]] + upper_ci[columns[i]], color='k', lw=0.5)
        ax.plot(time, mean_truth[columns[i]] + lower_ci[columns[i]], color='k', lw=0.5)
        ax.plot(time, mean_truth[columns[i]] - upper_ci[columns[i]], color='k', lw=0.5)
        ax.plot(time, mean_truth[columns[i]] - lower_ci[columns[i]], color='k', lw=0.5)

        ax.fill_between(time, mean_truth[columns[i]] + upper_ci[columns[i]],
                        mean_truth[columns[i]] + lower_ci[columns[i]], alpha=0.5, color='pink')
        ax.fill_between(time, mean_truth[columns[i]] - upper_ci[columns[i]],
                        mean_truth[columns[i]] - lower_ci[columns[i]], alpha=0.5, color='pink')
        ax.fill_between(time, mean_truth[columns[i]] - mean_err[columns[i]],
                        mean_truth[columns[i]] + mean_err[columns[i]], alpha=0.4, color='grey')

        ax.set_ylabel(columns[i], fontsize=16)
        ax.legend(loc=(0.82, 0.45), prop={'size': 16})

        if i == len(columns) - 1:
            ax.set_xlabel('Time [Days]', fontsize=20)
        fig.suptitle(
            f'Bootstapped {int(ci_level * 100)}% Confidence Intervals of MAE Estimates With Respect to Mean Data '
            f'- {species}', fontsize=24)
        plt.savefig(join(output_path, f'plots/{species}_MAE_data_{model_name}.png'), bbox_inches='tight')

    return


def crps_ens_bootstrap(truth, preds, columns, n_bs_samples=1000, ci_level=0.95):
    """
    Calculate Continuous Ranked Probability Score across validation experiments for ensemble
    Args:
        truth (pd.DataFrame): Modeled observations
        preds (pd.DataFrame): Predictions
        columns: Columns to calculate CRPS on
        n_bs_samples (int): Number of bootstrap resamples (defaults to 1000)
        ci_level (float): Confidence level in decimal form (defaults to 0.95)

    Returns:
        (Dicts): Mean CRPS, Lower CRPS CI, Upper CRPS CI
    """
    truth = truth.set_index(['member', 'id'])
    preds = preds.set_index(['member', 'id'])
    
    num_exps = truth.index.levels[1].nunique()
    time_steps = truth['Time [s]'].unique()
    min_quant = (1 - ci_level) / 2
    max_quant = 1 - min_quant
    all_crps, all_upper_ci, all_lower_ci = {}, {}, {}

    for phase in columns:

        crps, upper_ci, lower_ci = [], [], []

        for time_step in time_steps:

            bs_crps = []

            t_sub = truth[truth['Time [s]'] == time_step][phase].unstack('member')[0]
            p_sub = preds[preds['Time [s]'] == time_step][phase].unstack('member')
            crps_sub = ps.crps_ensemble(t_sub, p_sub)
            crps.append(crps_sub.mean())

            for _ in range(n_bs_samples):
                sample_crps = np.random.choice(crps_sub, num_exps, replace=True)
                bs_crps.append(sample_crps.mean())

            upper_ci.append(np.quantile(bs_crps, max_quant))
            lower_ci.append(np.quantile(bs_crps, min_quant))

        all_crps[phase] = crps
        all_upper_ci[phase] = upper_ci
        all_lower_ci[phase] = lower_ci

    return all_crps, all_lower_ci, all_upper_ci


def plot_crps_bootstrap(truth, preds, columns, output_path, species, model_name, n_bs_samples=1000, ci_level=0.95,
                        only_stable=True, stable_thresh=1):
    """
    Plot Continuous Ranked Probability Score across validation experiments for ensemble
    Args:
        truth (pd.DataFrame): Modeled observations
        preds (pd.DataFrame): Predictions
        columns: Columns to plot CRPS of
        output_path (str): Base path to save to
        species (str): Species name (used for titling and file naming)
        model_name (str): Model name (used for titleing and file naming)
        n_bs_samples (int): Number of bootstrap resamples (defaults to 1000)
        ci_level (float): Confidence level in decimal form (defaults to 0.95)
        only_stable (bool): If only stable experiments should be used
        stable_thresh (float): Magnitude to determine if experiments are stable (only used if only_stable=True)

    Returns:
    """
    if only_stable:
        stable_exps = get_stability(preds, stable_thresh, columns)[0]
        truth = truth[truth['id'].isin(stable_exps)]
        preds = preds[preds['id'].isin(stable_exps)]

    crps, lower_ci, upper_ci = crps_ens_bootstrap(truth, preds, columns, n_bs_samples, ci_level)
    time = truth['Time [s]'].unique() / 60 / 60 / 24

    fig, ax = plt.subplots(len(columns), 1, figsize=(22, 22), sharex=True)
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.subplots_adjust(top=0.95)

    for i, phase in enumerate(crps.keys()):
        if phase == 'Precursor [ug/m3]':
            ax[i].set_yscale('log')
        colors = ['r', 'g', 'b']
        ax[i].xaxis.set_tick_params(labelsize=16)
        ax[i].plot(time, crps[phase], color=colors[i], lw=3, label=f'Mean {phase} CRPS')
        ax[i].plot(time, lower_ci[phase], color='k', lw=1)
        ax[i].plot(time, upper_ci[phase], color='k', lw=1)
        ax[i].fill_between(time, lower_ci[phase], upper_ci[phase], alpha=0.2, color='grey')
        ax[i].set_ylabel(phase, fontsize=16)
        ax[i].legend(loc='upper right', prop={'size': 16})
        if i == len(columns) - 1:
            ax[i].set_xlabel('Time [Days]', fontsize=20)
        fig.suptitle(
            f'Bootstapped {int(ci_level * 100)}% Confidence Intervals of Ensembled CRPS - {species.title()}',
            fontsize=24)
        plt.savefig(join(output_path, f'plots/{species}_CRPS_{model_name}.png'), bbox_inches='tight')


def plot_unstability(preds, columns, output_path, model_name, stability_thresh=1):
    """
    Plot unstable runs by timestep for each mass phase specified
    Args:
        preds (pd.DataFrame): Predictions
        columns (list): Mass phases to be plotted
        output_path (str): Base output path
        model_name (str): Name of model plotted
        stability_thresh (float): Magnitude of threshold to determine instability (> thresh or < -thresh)

    Returns:
    """

    total_runs = int(len(preds) / preds['Time [s]'].nunique())
    time = preds['Time [s]'].unique() / 60 / 60 / 24
    colors = ['lightblue', 'red', 'green']
    plt.figure(figsize=(24, 8))
    plt.tick_params(axis='both', labelsize=18)

    for i, column in enumerate(columns):
        x = preds.groupby('Time [s]')[column].apply(
            lambda x: x[(x > stability_thresh) | (x < -stability_thresh)].count())
        sns.lineplot(time, x, color='k')
        plt.fill_between(time, x, color=colors[i], label=column)

    plt.legend(prop={'size': 16})
    plt.xlabel('Simulation Days', fontsize=20)
    plt.ylabel('Number Runaways', fontsize=20)
    plt.title(f'Ensemble Counts of Runaway Errors ({total_runs} Total Runs)', fontsize=24)
    plt.savefig(join(output_path, f'plots/unstable_{model_name}.png'), bbox_inches='tight')


def plot_scatter_analysis(preds, truth, train, val, cols, output_path, species, model_name, n_exps=10):
    """
    Produce scatter plot of each environmental feature with respect to mean mass for each phase. Include locations
    of top and bottom performing experiments.
    Args:
        preds: Box model predictions (df)
        truth: Box model truth (df)
        train: Training data (df)
        val: Validation data (df)
        cols: Features used to calculate MAE for performance comparison
        output_path: Output path (str)
        species: Modeled species
        model_name: Model name
        n_exps: Number of experiments to plot (best and worst performing)
    """
    mean_train = train.groupby('id').mean()
    mean_val = val.groupby('id').mean()
    best_exps, worst_exps = get_outliers(preds, truth, cols, n_exps)
    best_val = mean_val[mean_val.index.isin(best_exps)]
    worst_val = mean_val[mean_val.index.isin(worst_exps)]
    env_vars = ['temperature (K)', 'solar zenith angle (degree)', 'pre-existing aerosols (ug/m3)', 'o3 (ppb)',
                'nox (ppb)', 'oh (10^6 molec/cm3)']
    phase = ['Precursor [ug/m3]', 'Gas [ug/m3]', 'Aerosol [ug_m3]']

    fig, ax = plt.subplots(3, 6, figsize=(35, 20), constrained_layout=True)
    hue = 'temperature (K)'
    c = 'coolwarm'

    for i in range(3):

        for j, var in enumerate(env_vars):

            s = sns.scatterplot(data=mean_train, y=phase[i], x=var, hue=hue, ax=ax[i, j], palette=c, legend=False)
            s.set(xlabel=None)
            s.set(ylabel=None)

            u = sns.scatterplot(data=worst_val, y=phase[i], x=var, ax=ax[i, j], marker='X', s=300, color='k',
                                legend=False)
            u.set(xlabel=None)
            u.set(ylabel=None)

            w = sns.scatterplot(data=best_val, y=phase[i], x=var, ax=ax[i, j], marker='s', s=200, color='k',
                                legend=False)
            w.set(xlabel=None)
            w.set(ylabel=None)

            if i == 0:
                ax[i, j].set_title(var, fontsize=18, weight='bold')

            if j == 0:
                ax[i, j].set_ylabel(phase[i], fontsize=18, weight='bold')

    norm = plt.Normalize(mean_train[hue].min(), mean_train[hue].max())
    sm = plt.cm.ScalarMappable(cmap=c, norm=norm)
    cb = fig.colorbar(sm, ax=ax.flat, aspect=100)
    cb.set_label(label=hue, weight='bold', size=24)
    plt.suptitle(f'{species.capitalize()} - {model_name.upper()}', fontsize=50, y=1.08)
    plt.savefig(join(output_path, f'plots/scatter_analysis_{model_name}_{species}.png'), bbox_inches='tight')


def save_analysis_plots(all_truth, all_preds, train_input, val_input, output_path, output_vars, species, model_name):

    plot_ensemble(all_truth, all_preds, output_path, output_vars, species, model_name)
    all_truth.reset_index(inplace=True)
    all_preds.reset_index(inplace=True)
    plot_bootstrap_ci(all_truth, all_preds, output_vars, output_path, species, model_name)
    plot_crps_bootstrap(all_truth, all_preds, output_vars, output_path, species, model_name)
    plot_unstability(all_preds, output_vars, output_path, model_name)
    plot_scatter_analysis(all_preds, all_truth, train_input, val_input, output_vars,
                          output_path, species, model_name)
    
    
def fourier_analysis(preds, output_path, species, model_name):
    
    """
    Produce plots of the fourier transform of experiments in frequency space.
    All binned columns will be aggregated before performing the analysis.
    Args:
        preds: Box model predictions (df)
        output_path: Output path (str)
        species: Modeled species
        model_name: Model name
    https://stackoverflow.com/questions/52690632/analyzing-seasonality-of-google-trend-time-series-using-fft
    """
    # Average over the ensembles
    preds = preds.groupby(["id", "Time [s]"])
    preds = preds.mean()
    preds = preds.reset_index().drop(columns = ["member"])
    
    
    names = ["gas", "aerosol"]
    column_names = ["Gas [ug/m3]", "Aerosol [ug_m3]"]
    
    # Collapse binned data into aggregate quantities
    if column_names[0] not in preds.columns or column_names[1] not in preds.columns:
        preds[column_names[0]] = preds.loc[:, preds.columns.str.contains("Gas", regex=False)].sum(axis=1)
        preds[column_names[1]] = preds.loc[:, preds.columns.str.contains("Aerosol", regex=False)].sum(axis=1)
    preds = preds[['Time [s]', 'Precursor [ug/m3]', 'Gas [ug/m3]', 'Aerosol [ug_m3]', 'id']]
    
    val_exps = preds["id"].unique()
    bulk = np.zeros((len(val_exps), 719))

    plt.figure(figsize=(10, 10))

    for g, (name, column_name) in enumerate(zip(names, column_names)):

        for p, exp in enumerate(val_exps):

            c = (preds["id"] == exp)

            a_gtrend_orig = preds[c][column_name].copy()
            t_gtrend_orig = preds[c]["Time [s]"].copy()
            dt = t_gtrend_orig.iloc[1] - t_gtrend_orig.iloc[0]
            exp_id = exp.replace("Exp", "")

            a_gtrend_windowed = (a_gtrend_orig-np.median(a_gtrend_orig ))*tukey( len(a_gtrend_orig) )

            if exp == "Exp1601" or exp == "Exp1801":
                plt.subplot(3, 2, 1 + g)
                plt.plot( t_gtrend_orig, a_gtrend_orig, label=f'raw {name} data', c = "k"  )
                plt.plot( t_gtrend_orig, a_gtrend_windowed, label='windowed data', c = "r"  )
                plt.xlabel( 'secs' )
                plt.ylabel( column_name.replace("ug_m3", "ug/m3") )
                plt.title(f"Experiment {exp_id}")
                plt.legend()

            a_gtrend_psd = abs(rfft(a_gtrend_orig ))
            a_gtrend_psdtukey = abs(rfft(a_gtrend_windowed))
            a_gtrend_freqs = rfftfreq(len(a_gtrend_orig), d = dt)

            # For the PSD graph, we skip the first two points, this brings us more into a useful scale
            # those points represent the baseline (or mean), and are usually not relevant to the analysis
            if exp == "Exp1601" or exp == "Exp1801":
                plt.subplot(3, 2, 3 + g)
                plt.plot( a_gtrend_freqs[1:], a_gtrend_psd[1:], label=f'psd raw {name} data', c = "k"  )
                plt.plot( a_gtrend_freqs[1:], a_gtrend_psdtukey[1:], label='windowed psd', c = "r"  )
                plt.xlabel('frequency ($sec^{-1}$)')
                plt.ylabel('Amplitude')
                plt.xlim([0.0, 0.00005])
                plt.title(f"Experiment {exp_id}")
                plt.legend()

            bulk[p] = a_gtrend_psdtukey[1:]

        ave = np.mean(bulk, 0)
        plt.subplot(3, 2, 5 + g)
        plt.plot( a_gtrend_freqs[1:], ave, label='mean-windowed psd', c = "r", zorder = 2 )
        plt.xlabel('frequency ($sec^{-1}$)')
        plt.ylabel('Average Amplitude')
        plt.title("Average over 200 experiments")
        plt.xlim([0.0, 0.00005])

        period = 1. / a_gtrend_freqs[4:][np.argmax(ave[3:])]
        plt.plot([1./period, 1./period], [min(ave), max(ave)], 
                 c = 'b', ls = '--', zorder = 1, label = r"$T^{-1} =$ 24 hr")
        plt.legend()


    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"plots/{species}_fourier_analysis_{model_name}.pdf"), dpi = 300)
    plt.show()
