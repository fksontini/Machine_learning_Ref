#!/usr/bin/env python3
""" One hot encode """
import numpy as np


def one_hot_encode(Y, classes):
    """ One hot encode """
    try:
        return np.eye(classes)[Y.reshape(-1)].T
    except Exception:
        return None
