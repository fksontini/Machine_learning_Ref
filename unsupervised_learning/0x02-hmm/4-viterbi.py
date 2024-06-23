#!/usr/bin/env python3
"""Viterbi Algorithm"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """ Perform The Viterbi Algorithm """
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
    omega = np.zeros((N, T))
    prev = np.zeros((N, T))
    omega[:, 0] = Initial.T * Emission[:, Observation[0]]
    for t in range(1, T):
        for j in range(N):
            prob = omega[:, t - 1
                         ] * Emission[j, Observation[t]] * Transition[:, j]
            omega[j, t] = np.max(prob)
            prev[j, t] = np.argmax(prob, 0)
    P = np.max(omega[:, T - 1])
    S = []
    last_state = np.argmax(omega[:, T - 1])
    S.append(last_state)
    for i in range(T - 1, 0, -1):
        last_state = int(prev[last_state, i])
        S.append(last_state)
    return S[::-1], P
