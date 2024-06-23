#!/usr/bin/env python3
"""Multiply matrix"""


def mat_mul(mat1, mat2):
    """ Multiply Matrix """
    if len(mat1[0]) != len(mat2):
        return None
    m = []
    r = []
    tr = [[row[i] for row in mat2] for i in range(len(mat2[0]))]
    for i in mat1:
        for j in tr:
            r.append(sum([k * l for k, l in zip(i, j)]))
        m.append(r)
        r = []
    return m
