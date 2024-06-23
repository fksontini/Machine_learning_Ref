#!/usr/bin/env python3
""" Regularization """

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ Calculates the cost of a neural network with L2 regularization """
    sum = 0
    for i in range(1, L+1):
        sum += np.linalg.norm(weights["W" + str(i)])
    return cost + (lambtha / (2 * m)) * sum
