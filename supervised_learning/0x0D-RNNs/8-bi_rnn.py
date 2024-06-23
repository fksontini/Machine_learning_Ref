#!/usr/bin/env python3
""" Recurrent Neural Network """
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """ Perform forward propagation for a bidirectional RNN """
    t, m, _ = X.shape
    o = bi_cell.by.shape[1]
    h = h_0.shape[1]
    Hf = np.zeros((t + 1, m, h))
    Hb = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))
    Hf[0] = h_0
    Hb[-1] = h_t
    for i in range(1, t + 1):
        Hf[i] = bi_cell.forward(Hf[i - 1], X[i - 1])
    for i in reversed(range(0, t)):
        Hb[i] = bi_cell.backward(Hb[i + 1], X[i])
    H = np.concatenate((Hf[1:], Hb[:-1]), axis=-1)
    Y = bi_cell.output(H)
    return H, Y
