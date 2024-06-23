#!/usr/bin/env python3
"""Hidden Markov model"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """ Perform the forward algorithm for a hidden markov model """
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

    alpha = np.zeros((N, T))
    alpha[:, 0] = Initial.T * Emission[:, Observation[0]]
    for t in range(1, T):
        for j in range(N):
            alpha[j, t] = alpha[:, t - 1].dot(
                Transition[:, j]) * Emission[j, Observation[t]]
    P = np.sum(alpha[:, T - 1])
    return P, alpha
