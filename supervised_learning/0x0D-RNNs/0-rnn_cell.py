#!/usr/bin/env python3
""" Recurrent Neural Network """
import numpy as np


class RNNCell:
    """ Represent a cell of a simple RNN """
    def __init__(self, i, h, o):
        """ Initialization """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """Softmax Activation Function"""
        expo = np.exp(x - np.max(x))
        return expo / expo.sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """ Perform forward propagation for one time step """
        allinput = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(allinput, self.Wh) + self.bh)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, y
