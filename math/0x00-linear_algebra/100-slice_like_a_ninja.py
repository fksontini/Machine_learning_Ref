#!/usr/bin/env python3
""" Slice like a ninja"""


def np_slice(matrix, axes={}):
    """ Numpy slice matrix"""
    tmp = []
    result = []
    for i in range(len(matrix.shape)):
        if i in axes:
            tmp.append(slice(*axes[i]))
        else:
            tmp.append(slice(None))
    result = matrix[tuple(tmp)]
    return result
