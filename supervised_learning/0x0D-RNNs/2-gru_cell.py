#!/usr/bin/env python3
""" Recurrent Neural Network """
import numpy as np


class GRUCell:
    """ Represent a gated recurrent unit"""

    def __init__(self, i, h, o):
        """ initialization"""
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """Softmax Activation Function"""
        expo = np.exp(x - np.max(x))
        return expo / expo.sum(axis=1, keepdims=True)

    def sigmoid(self, x):
        """Sigmoid Activation Function"""
        return np.exp(-np.logaddexp(0, -x))

    def forward(self, h_prev, x_t):
        """ Perform forward propagation for one time step"""
        allinput = np.concatenate((h_prev, x_t), axis=1)
        resetgate = self.sigmoid(np.matmul(allinput, self.Wr) + self.br)
        updategate = self.sigmoid(np.matmul(allinput, self.Wz) + self.bz)

        xhr = np.hstack(((resetgate * h_prev), x_t))
        cav = np.tanh(np.dot(xhr, self.Wh) + self.bh)

        h_next = (1 - updategate) * h_prev + updategate * cav

        y = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(y)

        return h_next, y
