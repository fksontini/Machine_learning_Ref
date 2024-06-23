#!/usr/bin/env python3
""" Error analysis """

import numpy as np


def specificity(confusion):
    """ calculates the specificity for each class in a confusion matrix """
    truepos = confusion.diagonal()
    falsepos = confusion.sum(axis=0) - truepos
    falseneg = confusion.sum(axis=1) - truepos
    trueneg = confusion.sum() - (truepos + falsepos + falseneg)

    return trueneg / (trueneg + falsepos)
