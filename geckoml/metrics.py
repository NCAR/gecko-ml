from sklearn.metrics import confusion_matrix, mean_squared_error
import numpy as np


def calc_pdf_hist(x, x_bins):
    return np.histogram(x, x_bins, density=True)[0]


def hellinger(x, pdf_p, pdf_q):
    pdf_distances = (np.sqrt(pdf_p) - np.sqrt(pdf_q)) ** 2
    return np.trapz(pdf_distances, x) / 2


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def hellinger_distance(y_true, y_pred, bins=50):
    bin_points = np.linspace(np.minimum(y_true.min(), y_pred.min()),
                       np.maximum(y_true.max(), y_pred.max()),
                       bins)
    bin_centers = 0.5 * (bin_points[:-1] + bin_points[1:])
    y_true_pdf = calc_pdf_hist(y_true, bin_points)
    y_pred_pdf = calc_pdf_hist(y_pred, bin_points)
    return hellinger(bin_centers, y_true_pdf, y_pred_pdf)


def ensembled_box_metrics(y_true, y_pred):
    """ """
    y_true = y_true.sort_values(['id', 'Time [s]'], ascending=True).iloc[:, 1:-1]
    y_pred = y_pred.sort_values(['id', 'Time [s]'], ascending=True).iloc[:, :-2]

    hd = hellinger_distance(y_true.iloc[:,1], y_pred.iloc[:,1])
    rmse = root_mean_squared_error(y_true.iloc[:,1], y_pred.iloc[:,1])

    return hd, rmse


def ensembled_base_metrics(y_true, y_pred):
    """ """

    y_true = y_true.iloc[:,1:-1].values

    hd = hellinger_distance(y_true[:,1], y_pred[:,1])
    rmse = root_mean_squared_error(y_true[:,1], y_pred[:,1])

    return hd, rmse


def mae_time_series(y_true, y_pred, output_cols):

    columns = output_cols[1:-1]
    df_diff = np.abs(y_true[columns] - y_pred[columns])
    df_diff['Time [s]'] = y_true['Time [s]']
    mae = df_diff.groupby('Time [s]').mean()

    return mae


def match_true_exps(truth, preds, num_timesteps):

    exps = preds['id'].unique()
    true_sub = truth.loc[truth['id'].isin(exps)]
    true_sub = true_sub.groupby('id').apply(lambda x: x.iloc[:num_timesteps, :]).reset_index(drop=True)

    true_sub = true_sub.sort_values(['id', 'Time [s]'])
    preds = preds.sort_values(['id', 'Time [s]'])

    return true_sub, preds