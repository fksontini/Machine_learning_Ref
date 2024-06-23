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


def intersection(x, n, P, Pr):
    """ Calculate intersection """
    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if (P > 1).any() or (P < 0).any():
        raise ValueError('All values in P must be in the range [0, 1]')
    if (Pr > 1).any() or (Pr < 0).any():
        raise ValueError('All values in Pr must be in the range [0, 1]')
    if not np.isclose(np.sum(Pr), [1]):
        raise ValueError("Pr must sum to 1")
    return (likelihood(x, n, P) * Pr)


def marginal(x, n, P, Pr):
    """ Marginal probability """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if not ((0 <= P).all() and (P <= 1).all()):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not ((0 <= Pr).all() and (Pr <= 1).all()):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), [1]):
        raise ValueError("Pr must sum to 1")
    return np.sum(intersection(x, n, P, Pr))
