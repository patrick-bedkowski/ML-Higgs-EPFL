import numpy as np
import csv

def get_data(file_path):
    """
    Returns:
    """
    with open(file_path, 'r') as file_handle:
        data = np.genfromtxt(file_handle, delimiter=',', skip_header=1, dtype=object)

        # extract id and predictions from data
        ids = data[:, :1].astype(int)
        predictions = data[:, 1:2].astype(str)
        # encode class predictions to strings
        predictions = np.unique(predictions, return_inverse=True)[1]

        # skip ID and prediction
        data = data[:, 2:].astype(float)
        file_handle.close()

    with open(file_path, 'r') as file_handle:
        features = np.genfromtxt(file_handle, delimiter=',', max_rows=1, dtype=str)
        features = features[2:]
        file_handle.close()

    return ids, predictions, data, features

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})