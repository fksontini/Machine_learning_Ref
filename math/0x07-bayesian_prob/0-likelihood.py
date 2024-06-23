#!/usr/bin/env python3
""" Bayesian Probability """
import numpy as np


def likelihood(x, n, P):
    """Calculate likelihood"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if (P > 1).any() or (P < 0).any():
        raise ValueError('All values in P must be in the range [0, 1]')
    fact = np.math.factorial(n) / (np.math.factorial(x) *
                                   np.math.factorial(n - x))
    like = fact * (P ** x) * ((1 - P) ** (n - x))
    return like
