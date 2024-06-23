#!/usr/bin/env python3
"""Concat matrix"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ concat matrix """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [x.copy() for x in mat1] + [x.copy() for x in mat2]
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [x + y for x, y in zip(mat1, mat2)]
