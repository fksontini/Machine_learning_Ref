#!/usr/bin/env python3
"""K-means"""
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


def kmeans(X, k, iterations=1000):
    """performs K-means on a dataset"""
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    centroids = initialize(X, k)
    if centroids is None:
        return None, None
    dist = np.linalg.norm((X[:, np.newaxis, :] - centroids), axis=2)
    clusters = dist.argmin(1)
    for _ in range(iterations):
        cc = np.copy(centroids)
        for j in range(k):
            if X[clusters == j].size == 0:
                centroids[j] = initialize(X, 1)
            else:
                centroids[j] = X[clusters == j].mean(0)
        dist = np.linalg.norm((X[:, np.newaxis, :] - centroids), axis=2)
        clusters = dist.argmin(1)
        if (cc == centroids).all():
            break
    return centroids, clusters
