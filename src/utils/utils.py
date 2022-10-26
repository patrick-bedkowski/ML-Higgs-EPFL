import numpy as np


def replace_with_nans(arr):
    arr_cp = arr.copy()
    arr_cp[arr_cp == -999] = np.nan
    return arr_cp


def drop_nan_columns_above_treshold(arr):
    """
    :return: Returns dataframe without columns where number of nan values exceeded threshold. Indices of dropped columns
    """
    NANS_COL_TRESHOLD = 0.7
    arr_nans_ratio = np.count_nonzero(np.isnan(arr), axis=0)/arr.shape[0]
    col_indices_above_threshold = np.argwhere(arr_nans_ratio >= NANS_COL_TRESHOLD)
    arr_deleted = _drop_columns(arr, col_indices_above_threshold)
    return arr_deleted, col_indices_above_threshold


def _drop_columns(arr, indices):
    return np.delete(arr, indices, axis=1)


def fill_values_mean(arr):
    arr_cp = arr.copy()

    col_mean = np.nanmean(arr_cp, axis=0)
    inds = np.where(np.isnan(arr_cp))
    arr_cp[inds] = np.take(col_mean, inds[1])
    return arr_cp


def fill_values_median(arr):
    arr_cp = arr.copy()

    col_median = np.nanmedian(arr_cp, axis=0)
    inds = np.where(np.isnan(arr_cp))
    arr_cp[inds] = np.take(col_median, inds[1])
    return arr_cp


def iqr_rule(arr: np.array, sensitivity: float = 1.5):
    """
    Removes outliers according to IQR rule
    :param arr: Array
    :param sensitivity: Sensitivity of data removal. 1.5 works well for Gaussian distributed data
    """
    lower_limit, upper_limit = get_confidence_interval(arr, sensitivity)

    arr_cleaned = np.delete(arr, np.where((arr < lower_limit) | (arr > upper_limit)), axis=0)
    return arr_cleaned


def get_confidence_interval(arr, sensitivity):
    q75, q25 = np.percentile(arr, [75, 25], axis=0)
    iqr = q75 - q25

    lower_limit = q25 - sensitivity*iqr
    upper_limit = q75 + sensitivity*iqr
    return lower_limit, upper_limit
