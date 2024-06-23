#!/usr/bin/env python3
""" Normalization function """

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ creates the training operation for a neural network """
    return alpha / (1 + (decay_rate * np.floor(global_step / decay_step)))
