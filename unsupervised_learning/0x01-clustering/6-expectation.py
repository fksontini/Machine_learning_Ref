#!/usr/bin/env python3
""" Expectation """
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """ Calculate the expectation step in the EM algorithm for a GMM """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if X.shape[1] != S.shape[1] or S.shape[1] != S.shape[2]:
        return (None, None)
    if X.shape[1] != m.shape[1] or m.shape[0] != S.shape[0]:
        return (None, None)
    if pi.shape[0] != m.shape[0]:
        return (None, None)
    if not np.isclose(np.sum(pi), 1):
        return None, None

    k = pi.shape[0]
    P = [pdf(X, m[i], S[i]) * pi[i] for i in range(k)]
    g = P / np.sum(P, axis=0)
    likelihood = np.sum(np.log(np.sum(P, axis=0)))
    return g, likelihood
