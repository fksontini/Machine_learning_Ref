#!/usr/bin/env python3
""" Probability Density Function"""
import numpy as np


def pdf(X, m, S):
    """Calculate the probability density function of a Gaussian distribution"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None
    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[1] != S.shape[0]:
        return None
    _, d = X.shape
    if d != m.shape[0] or (d, d) != S.shape:
        return None
    A = (X - m)
    x = np.exp(- np.dot(np.dot(A, np.linalg.inv(S)), A.T).diagonal() / 2)
    y = np.sqrt((2 * np.pi) ** d * np.linalg.det(S))
    PDF = np.where((x / y) < 1e-300, 1e-300, (x / y))
    return PDF
