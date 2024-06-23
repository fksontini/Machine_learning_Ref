#!/usr/bin/env python3
"""Transpose of a matrix"""


def matrix_transpose(matrix):
    """ transpose of a 2D matrix"""
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
