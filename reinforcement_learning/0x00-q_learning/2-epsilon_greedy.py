#!/usr/bin/env python3
""" Q Learning """
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """ Use epsilon-greedy to determine the next action """
    action = 0
    p = np.random.uniform(0, 1)
    if p < epsilon:
        action = np.random.randint(4)
    else:
        action = np.argmax(Q[state, :])
    return action
