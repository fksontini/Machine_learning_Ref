#!/usr/bin/env python3
"""Initialize GMM """
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """ Initialize variables for a Gaussian Mixture Model"""
    try:
        _, d = X.shape
        m, _ = kmeans(X, k)
        pi = np.full(k, 1 / k)
        ident = [np.identity(d)]
        S = np.array(ident * k)
        return pi, m, S
    except Exception:
        return None, None, None
