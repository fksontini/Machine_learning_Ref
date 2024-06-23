#!/usr/bin/env python3
""" Regularization """

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout"""
    cache = {"A0": X}
    for i in range(L):
        z = np.matmul(weights["W" + str(i + 1)],
                      cache["A" + str(i)]) + weights["b" + str(i + 1)]
        drop = np.random.binomial(1, keep_prob, size=z.shape)
        if i == L - 1:
            x = np.exp(z)
            cache["A" + str(i + 1)] = x / np.sum(x, axis=0, keepdims=True)
        else:
            cache["A" + str(i + 1)] = np.tanh(z)
            cache["D" + str(i + 1)] = drop
            cache["A" + str(i + 1)] = (cache["A" + str(i + 1)]
                                       ) * cache[
                "D" + str(i + 1)] / keep_prob
    return cache
