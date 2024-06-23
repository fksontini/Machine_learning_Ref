#!/usr/bin/env python3
""" Agglomerative """
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """ Perform agglomerative clustering on a dataset """
    hierarchy = scipy.cluster.hierarchy
    linkage = hierarchy.linkage(X, method='ward')
    clss = hierarchy.fcluster(linkage, dist, criterion='distance')
    hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.figure()
    plt.show()
    return clss
