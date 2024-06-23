#!/usr/bin/env python3
""" Error analysis """

import numpy as np


def precision(confusion):
    """ calculates the precision for each class in a confusion matrix """
    return confusion.diagonal() / confusion.sum(axis=0)
