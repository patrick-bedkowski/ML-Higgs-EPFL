import numpy as np

TRAIN_FILEPATH, TEST_FILEPATH = "../data/train.csv", "../data/test.csv"


def get_train_higgs_data():
    """
    Returns:
    """
    with open(TRAIN_FILEPATH, 'r') as file_handle:
        data = np.genfromtxt(file_handle, delimiter=',', skip_header=1, dtype=object)

        # extract id and predictions from data
        ids = data[:, :1].astype(int)
        predictions = data[:, 1:2].astype(str)
        # encode class predictions to strings
        predictions = np.unique(predictions, return_inverse=True)[1]

        # skip ID and prediction
        data = data[:, 2:].astype(float)
        file_handle.close()

    with open(TRAIN_FILEPATH, 'r') as file_handle:
        features = np.genfromtxt(file_handle, delimiter=',', max_rows=1, dtype=str)
        features = features[2:]
        file_handle.close()

    return ids, predictions, data, features


def get_test_higgs_data():
    """
    Returns:
    """
    with open(TEST_FILEPATH, 'r') as file_handle:
        data = np.genfromtxt(file_handle, delimiter=',', skip_header=1, dtype=object)

        # extract id and predictions from data
        ids = data[:, :1].astype(int)
        predictions = data[:, 1:2].astype(str)
        # encode class predictions to strings
        predictions = np.unique(predictions, return_inverse=True)[1]

        # skip ID and prediction
        data = data[:, 2:].astype(float)
        file_handle.close()

    with open(TEST_FILEPATH, 'r') as file_handle:
        features = np.genfromtxt(file_handle, delimiter=',', max_rows=1, dtype=str)
        features = features[2:]
        file_handle.close()

    return ids, predictions, data, features
#%%
