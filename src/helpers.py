# -*- coding: utf-8 -*-
"""some helper functions."""

import numpy as np
import csv


def normalize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    std_x = np.std(x)
    x = (x - mean_x)/ std_x
    return x, np.mean(x), np.std(x)

#
# def standardize(x):
#     """Standardize the original data set."""
#     mean_x = np.mean(x)
#     x = x - mean_x
#     std_x = np.std(x)
#     x = x / std_x
#     return x, mean_x, std_x


def build_model_data(y, x):
    bias = np.ones(len(x))
    tx = np.c_[bias, x]
    return y, tx


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1),'Prediction':int(r2)})
