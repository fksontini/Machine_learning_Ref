#!/usr/bin/env python3
""" Maximization"""
import numpy as np


def maximization(X, g):
    """ Calculate the maximization step in the EM algorithm for a GMM """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None
    if not np.all(np.isclose(g.sum(axis=0), 1)):
        return None, None, None
    n, _ = X.shape
    post_sum = g.sum(axis=1)
    pi = post_sum / n
    m = np.dot(g, X) / post_sum[:, np.newaxis]
    S = np.zeros((m.shape[0], m.shape[1], m.shape[1]))
    for i in range(g.shape[0]):
        diff = X - m[i]
        S[i] = np.dot((diff * g[i, :, np.newaxis]).T, diff) / post_sum[i]

    return pi, m, S
