#!/usr/bin/env python3
""" Normalization function """

import numpy as np


def normalization_constants(X):
    """ calculates the normalization constants of a matrix """

    return np.mean(X, axis=0), np.std(X, axis=0)
