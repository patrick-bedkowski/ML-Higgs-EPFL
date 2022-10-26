import numpy as np


def least_squares(y, tx):
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = compute_loss(y, tx, w)
    return w, mse


def compute_mse(e):
    """Calculate the mse for vector e"""
    return 1/2*np.mean(e**2)


def compute_loss(y, tx, w):
    """Calculate the mse loss"""
    e = y - tx.dot(w)
    return compute_mse(e)
