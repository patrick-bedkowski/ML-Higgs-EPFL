import numpy as np

def mse(y, x, w):
    err = y - x.dot(w)
    mse = err.dot(err) / (2 * len(y))

    return mse

def ridge_regression(y, tx, lambda_):
    LAMBDA = 2 * len(y) * lambda_
    
    # Solve Normal Equations with Regularization Term
    w = np.linalg.solve((tx.T @ tx + (np.eye(len(tx[1])) * LAMBDA)), tx.T @ y)
    
    return w