#!/usr/bin/env python3
"""Variance"""
import numpy as np


def variance(X, C):
    """ calculate the total intra-cluster variance for a data set """
    try:
        dist = np.linalg.norm((X[:, np.newaxis, :] - C), axis=2)
        clusters = dist.argmin(1)
        var = np.linalg.norm(X - C[clusters]) ** 2
        return var
    except Exception:
        return None
