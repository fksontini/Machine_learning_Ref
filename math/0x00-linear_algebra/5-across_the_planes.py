#!/usr/bin/env python3
"""sum matrix"""


def add_matrices2D(mat1, mat2):
    """ sum matrix """

    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [[x + y for x, y in zip(mat1[i], mat2[i])]
            for i in range(len(mat1))]
