#!/usr/bin/env python3
""" Regular Markov chain """

import numpy as np


def markov_chain(P, s, t=1):
    """ Determine the probability of a markov chain being
    in a particular state after a specified number of iterations """
    # if type(P) is not np.ndarray or len(P.shape) != 2:
    #     return None
    # n = P.shape[1]
    # if type(s) is not np.ndarray or s.shape != (1, n):
    #     return None
    # if np.any(np.sum(P, axis=1)) != 1:
    #     return None
    for _ in range(t):
        s = np.matmul(s, P)
    return s
