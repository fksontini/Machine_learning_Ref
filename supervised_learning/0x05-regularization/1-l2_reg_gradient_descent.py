#!/usr/bin/env python3
""" Regularization """

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates the weights and biases of a neural network """
    m = Y.shape[1]
    dz = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        tanh = cache["A" + str(i - 1)]
        dw = np.matmul(dz, tanh.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        dtanh = 1 - tanh * tanh
        dz = np.matmul(weights["W" + str(i)].T, dz) * dtanh

        reg = dw + (lambtha / m) * weights["W" + str(i)]
        weights["W" + str(i)] -= alpha * reg
        weights["b" + str(i)] -= alpha * db
