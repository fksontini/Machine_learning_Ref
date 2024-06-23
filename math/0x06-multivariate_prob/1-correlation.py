#!/usr/bin/env python3
""" Multivariate probability"""

import numpy as np


def correlation(C):
    """ Correlation Matrix """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    diag = np.diag(np.sqrt(np.diag(C)))
    invd = np.linalg.inv(diag)
    corr = np.dot(np.dot(invd, C), invd)
    return corr
