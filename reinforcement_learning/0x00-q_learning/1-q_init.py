#!/usr/bin/env python3
""" Q Learning """
import numpy as np


def q_init(env):
    """ Initialize the Q-table """
    return np.zeros([env.observation_space.n, env.action_space.n])
