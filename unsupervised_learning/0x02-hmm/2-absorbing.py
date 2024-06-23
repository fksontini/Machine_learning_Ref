#!/usr/bin/env python3
""" Absorbing Markov Chain """
import numpy as np


def absorbing(P):
    """ Determine if a markov chain is absorbing """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    n, m = P.shape
    if n != m:
        return None
    if np.any(P < 0):
        return None
    if not np.all(np.isclose(P.sum(axis=1), 1)):
        return None
    return np.any(P.diagonal() == 1)
