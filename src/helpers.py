import numpy as np
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
    
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.
        
    Returns:
        poly: numpy array of shape (N,d+1)
    """
    # This function should return the matrix formed
    # by applying the polynomial basis to the input data
    tx = np.c_[np.ones(len(x)), x]
    if(degree >= 2):
        for i in range(2, degree+1):
            tx = np.c_[tx, np.power(x, i)]
    return tx