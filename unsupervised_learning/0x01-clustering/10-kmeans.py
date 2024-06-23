#!/usr/bin/env python3
""" K-means """
import sklearn.cluster


def kmeans(X, k):
    """ Perform K-means on a dataset """
    model = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    clss = model.labels_
    C = model.cluster_centers_
    return C, clss
