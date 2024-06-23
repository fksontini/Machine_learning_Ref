#!/usr/bin/env python3
""" Normalization function """

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """ creates the training operation for a neural network """
    v = beta1 * v + (1 - beta1) * grad
    v_corrected = v / (1 - beta1 ** t)

    s = beta2 * s + (1 - beta2) * (grad ** 2)
    s_corrected = s / (1 - beta2 ** t)

    var = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)
    return var, v, s
