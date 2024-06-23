#!/usr/bin/env python3
"""  Multivariance Probability """
import numpy as np


def mean_cov(X):
    """ Mean and covariance """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    if X.shape[0] < 2:
        raise ValueError('X must contain multiple data points')
    n = X.shape[0]
    mean = X.mean(axis=0, keepdims=True)
    arg_mean = X - mean
    cov = np.matmul(arg_mean.T, arg_mean) / (n - 1)
    return mean, cov
