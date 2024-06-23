#!/usr/bin/env python3
""" advanced linear algebra """


def determinant(matrix):
    """ Determinant of a matrix """
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')
    if any(not isinstance(i, list) for i in matrix):
        raise TypeError('matrix must be a list of lists')
    if matrix == [[]]:
        return 1
    if any(len(i) != len(matrix) for i in matrix):
        raise ValueError('matrix must be a square matrix')
    n = len(matrix)
    if matrix == [[]]:
        return 1
    if n == 1:
        return matrix[0][0]
    else:
        det = 0
        for i in range(n):
            extract = []
            for j in range(1, n):
                row = []
                for k in range(n):
                    if k != i:
                        row.append(matrix[j][k])
                extract.append(row)
            det += (-1) ** i * matrix[0][i] * determinant(extract)
    return det
