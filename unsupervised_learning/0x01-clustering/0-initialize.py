#!/usr/bin/env python3
""" Clustering """
import numpy as np


def initialize(X, k):
    """ Initializes cluster centroids for K-mean"""
    if not isinstance(k, int) or k <= 0:
        return None
    try:
        _, d = X.shape
        centroids = np.random.uniform(
            low=np.min(
                X, axis=0), high=np.max(
                X, axis=0), size=(
                k, d))
    except Exception:
        return None
    return centroids
