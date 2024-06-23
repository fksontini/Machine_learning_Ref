#!/usr/bin/env python3
""" Dimensionality Reduction"""

import numpy as np


def pca(X, ndim):
    """ Principal component analysis """
    Xcentered = X - np.mean(X, axis=0)
    U, S, V = np.linalg.svd(Xcentered)
    T = np.matmul(Xcentered, V.T[:, :ndim])
    return T
