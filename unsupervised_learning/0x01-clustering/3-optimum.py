#!/usr/bin/env python3
"""Optimize k """
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """test for the optimum number of clusters by variance"""
    try:
        if kmax is None:
            kmax = X.shape[0]
        if not isinstance(kmin, int) or kmin < 1:
            return None, None
        if not isinstance(kmax, int) or kmin >= kmax:
            return None, None
        results = []
        d_vars = []
        for k in range(kmin, kmax + 1):
            centroid, cluster = kmeans(X, k, iterations)
            if k == kmin:
                smallest_var = variance(X, centroid)
            results.append((centroid, cluster))
            var = variance(X, centroid)
            d_vars.append(smallest_var - var)
    except Exception:
        return None, None
    return results, d_vars
