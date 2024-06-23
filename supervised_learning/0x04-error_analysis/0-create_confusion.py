#!/usr/bin/env python3
""" Error analysis """

import numpy as np


def create_confusion_matrix(labels, logits):
    """ creates a confusion matrix """
    m, classes = labels.shape
    cfn = np.zeros((classes, classes))
    for k in range(m):
        i = np.nonzero(labels[k, :])
        j = np.nonzero(logits[k, :])
        cfn[i, j] += 1
    return cfn
