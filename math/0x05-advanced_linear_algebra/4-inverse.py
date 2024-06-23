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


def minor(matrix):
    """ minor of a matrix """
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')
    if any(not isinstance(i, list) for i in matrix):
        raise TypeError('matrix must be a list of lists')
    if any(len(i) != len(matrix) for i in matrix):
        raise ValueError('matrix must be a non-empty square matrix')
    if len(matrix[0]) == 0:
        raise ValueError('matrix must be a non-empty square matrix')
    n = len(matrix)
    if n == 1:
        return [[1]]
    q = []
    for i in range(n):
        a = []
        for j in range(n):
            a.append(determinant([row[:j] + row[j + 1:]
                                  for row in (matrix[:i] + matrix[i + 1:])]))
        q.append(a)
    return q


def cofactor(matrix):
    """ cofactor of a matrix """
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')
    if any(not isinstance(i, list) for i in matrix):
        raise TypeError('matrix must be a list of lists')
    if any(len(i) != len(matrix) for i in matrix):
        raise ValueError('matrix must be a non-empty square matrix')
    if len(matrix[0]) == 0:
        raise ValueError('matrix must be a non-empty square matrix')
    if len(matrix) == 1:
        return [[1]]
    mm = minor(matrix)
    for i in range(len(mm)):
        for j in range(len(mm)):
            sign = (-1) ** (i + j)
            mm[i][j] *= sign
    return mm


def adjugate(matrix):
    """ adjugate of a matrix """
    cm = cofactor(matrix)
    return [list(i) for i in zip(*cm)]


def inverse(matrix):
    """ inverse of a matrix """
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')
    if any(not isinstance(i, list) for i in matrix):
        raise TypeError('matrix must be a list of lists')
    if any(len(i) != len(matrix) for i in matrix):
        raise ValueError('matrix must be a non-empty square matrix')
    if len(matrix[0]) == 0:
        raise ValueError('matrix must be a non-empty square matrix')
    det = determinant(matrix)
    am = adjugate(matrix)
    if det == 0:
        return None
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            am[i][j] /= det
    return am
