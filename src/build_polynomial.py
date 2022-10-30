# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    poly = np.zeros((len(x), degree + 1))
    
    for n in range(len(x)):
        for d in range(degree + 1):
            if d == 0:
                poly[n,d] = 1
            else:
                poly[n,d] = x[n]**d
    
    return poly
    # ***************************************************
    raise NotImplementedError