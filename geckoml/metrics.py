from sklearn.metrics import mean_squared_error
import numpy as np


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


def ensembled_box_metrics(y_true, y_pred):
    """ Call a variety of metrics to be calculated (Hellenger distance and RMSE, currently) on Box emulator results
    Args:
        y_true (np.array): True output data
        y_pred (np.array): Predicted output data
    """
    y_true = y_true.sort_values(['id', 'Time [s]'], ascending=True).iloc[:, 1:-1]
    y_pred = y_pred.sort_values(['id', 'Time [s]'], ascending=True).iloc[:, :-2]

    hd = hellinger_distance(y_true.iloc[:, 1], y_pred.iloc[:, 1])
    rmse = root_mean_squared_error(y_true.iloc[:, 1], y_pred.iloc[:, 1])

    return hd, rmse


def ensembled_base_metrics(y_true, y_pred):
    """ Call a variety of metrics to be calculated (Hellenger distance and RMSE, currently) on Base Model results
    Args:
        y_true (np.array): True output data
        y_pred (np.array): Predicted output data
    """

    y_true = y_true.iloc[:, 1:-1].values

    hd = hellinger_distance(y_true[:, 1], y_pred[:, 1])
    rmse = root_mean_squared_error(y_true[:, 1], y_pred[:, 1])

    return hd, rmse


def mae_time_series(y_true, y_pred):
    """ Calculate the Mean Absolute Error across timesteps
    Args:
        y_true (np.array): True output data
        y_pred (np.array): Predicted output data
    """
    time_series = y_true['Time [s]']
    y_true = y_true.sort_values(['id', 'Time [s]'], ascending=True).iloc[:, 1:-1]
    y_pred = y_pred.sort_values(['id', 'Time [s]'], ascending=True).iloc[:, :-2]

    df_diff = np.abs(y_true - y_pred)

    df_diff['Time [s]'] = time_series
    mae = df_diff.groupby('Time [s]').mean()

    return mae


def match_true_exps(truth, preds, num_timesteps):
    """ Retrieve true values that match the experiments used in previous box emulator run
    Args:
        truth (DataFrame): True output data
        preds (DataFrame): Predicted output data
        num_timesteps (int): number of timesteps used in emulator runs to pull equivalent number from true data
    """
    exps = preds['id'].unique()
    true_sub = truth.loc[truth['id'].isin(exps)]
    true_sub = true_sub.groupby('id').apply(lambda x: x.iloc[:num_timesteps, :]).reset_index(drop=True)

    true_sub = true_sub.sort_values(['id', 'Time [s]'])
    preds = preds.sort_values(['id', 'Time [s]'])

    return true_sub, preds
