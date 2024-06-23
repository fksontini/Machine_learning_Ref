#!/usr/bin/env python3
""" One hot decode """
import numpy as np


def one_hot_decode(one_hot):
    """ One hot decode """
    if not isinstance(one_hot, np.ndarray):
        return None
    if len(one_hot.shape) != 2:
        return None
    return np.argmax(one_hot.T, axis=1)
