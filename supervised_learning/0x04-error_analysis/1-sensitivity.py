#!/usr/bin/env python3
""" Error analysis """

import numpy as np


def sensitivity(confusion):
    """ calculates the sensitivity for each class in a confusion matrix """
    return confusion.diagonal() / confusion.sum(axis=1)
