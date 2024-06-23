#!/usr/bin/env python3
"""Function for matrix operations"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concat function"""
    return np.concatenate((mat1, mat2), axis=axis)
