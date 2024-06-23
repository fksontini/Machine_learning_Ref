#!/usr/bin/env python3
""" Normalization function """

import numpy as np


def shuffle_data(X, Y):
    """ shuffles the data points in two matrices the same way """
    P = np.random.permutation(X.shape[0])
    return X[P], Y[P]
