#!/usr/bin/env python3
""" Regular Markov Chain """
import numpy as np


def regular(P):
    """ Determine the steady state probabilities of a regular markov chain """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    n, m = P.shape
    if n != m:
        return None
    if np.any(P <= 0):
        return None
    if not np.all(np.isclose(P.sum(axis=1), 1)):
        return None

    evals, evecs = np.linalg.eig(P.T)
    evecs = evecs[:, np.isclose(evals, 1)]
    steady = evecs / evecs.sum()
    return steady.T
