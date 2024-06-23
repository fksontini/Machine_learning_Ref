#!/usr/bin/env python3
"""Function for matrix operations"""


def np_elementwise(mat1, mat2):
    """Elementwise operations"""
    s = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return s, sub, mul, div
