#!/usr/bin/env python3
""" Normalization function """

import numpy as np


def moving_average(data, beta):
    """ calculates the weighted moving average of a data set """
    ema = []
    avg = 0
    for i in range(len(data)):
        avg = beta*avg + (1 - beta)*data[i]
        bc = avg/(1 - beta**(i+1))
        ema.append(bc)
    return ema
