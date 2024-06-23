#!/usr/bin/env python3
""" Regularization """

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates the weights of a neural network with Dropout"""
    m = Y.shape[1]
    dz = cache["A" + str(L)] - Y
    for j in range(L, 0, -1):
        tanh = cache["A" + str(j - 1)]
        dW = np.matmul(dz, tanh.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        dtanh = 1 - tanh * tanh
        dz = np.matmul(weights[
            "W" + str(j)].T, dz) * dtanh
        if j > 1:
            dz *= cache["D" + str(j - 1)]
            dz /= keep_prob

        weights["W" + str(j)] -= alpha * dW
        weights["b" + str(j)] -= alpha * db
