import numpy as np

def ridge_regression(y, tx, lambda_):
    LAMBDA = 2 * len(y) * lambda_
    
    # Solve Normal Equations with Regularization Term
    w = np.linalg.solve((tx.T @ tx + (np.eye(len(tx[1])) * LAMBDA)), tx.T @ y)
    
    return w