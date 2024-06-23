#!/usr/bin/env python3
""" Normalization function """

import numpy as np


def normalize(X, m, s):
    """ normalizes (standardizes) a matrix """
    X = X - m
    X = X / s
    return X
