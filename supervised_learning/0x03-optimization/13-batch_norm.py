#!/usr/bin/env python3
""" Normalization function """

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ creates the training operation for a neural network """
    m = np.mean(Z, axis=0)
    s = np.std(Z, axis=0)
    normalized = (Z - m) / np.sqrt(s ** 2 + epsilon)
    batch = gamma * normalized + beta
    return batch
