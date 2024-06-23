#!/usr/bin/env python3
""" Dimensionality Reduction"""

import numpy as np


def pca(X, var=0.95):
    """ Principal component analysis """
    U, S, V = np.linalg.svd(X)
    M = np.cumsum(S) / np.sum(S)
    i = np.where(M <= var, 1, 0)
    i = np.sum(i)
    return V.T[:, :i + 1]
