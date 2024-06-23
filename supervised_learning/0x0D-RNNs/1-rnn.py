#!/usr/bin/env python3
""" Recurrent Neural Network """
import numpy as np


def rnn(rnn_cell, X, h_0):
    """ Perform forward propagation for a simple RNN """
    t, m, i = X.shape
    m, h = h_0.shape
    o = rnn_cell.Wy.shape[1]
    hiddenstates = np.zeros(shape=(t + 1, m, h))
    output = np.zeros(shape=(t, m, o))
    hiddenstates[0] = h_0
    for s in range(t):
        h_0, y = rnn_cell.forward(h_0, X[s, :, :])
        hiddenstates[s + 1, :, :] = h_0
        output[s, :, :] = y
    return hiddenstates, output
