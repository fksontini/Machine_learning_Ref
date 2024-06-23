#!/usr/bin/env python3
"""Hidden Markov model"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """ Perform the backward algorithm for a hidden markov model """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    T, = Observation.shape
    N, M = Emission.shape
    if not isinstance(Transition, np.ndarray) or Transition.shape != (N, N):
        return None, None
    if np.any(np.sum(Transition, axis=1) != 1):
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.shape != (N, 1):
        return None, None
    if np.sum(Initial) != 1:
        return None, None

    beta = np.zeros((N, T))
    beta[:, T - 1] = np.ones(N)
    for t in range(T - 2, -1, -1):
        for j in range(N):
            beta[j, t] = (beta[:, t + 1] * Emission[:, Observation[t + 1]]
                          ).dot(Transition[j, :])
    P = np.sum(beta[:, 0] * Emission[:, Observation[0]] * Initial.T)
    return P, beta
