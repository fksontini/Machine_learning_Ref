#!/usr/bin/env python3
""" Normalization function """

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ updates a variable using the gradient descent """
    v = v * beta1 + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
