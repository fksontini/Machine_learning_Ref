#!/usr/bin/env python3
""" Recurrent Neural Network """
import numpy as np


class LSTMCell:
    """ Represent an LSTM unit """

    def __init__(self, i, h, o):
        """ initialization"""

        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """Softmax Activation Function"""
        expo = np.exp(x - np.max(x))
        return expo / expo.sum(axis=1, keepdims=True)

    def sigmoid(self, x):
        """Sigmoid Activation Function"""
        return np.exp(-np.logaddexp(0, -x))

    def forward(self, h_prev, c_prev, x_t):
        """ Perform forward propagation for one time step """
        xh = np.concatenate((h_prev, x_t), axis=1)
        fgo = self.sigmoid(np.matmul(xh, self.Wf) + self.bf)
        igo = self.sigmoid(np.matmul(xh, self.Wu) + self.bu)
        candidate = np.tanh(np.matmul(xh, self.Wc) + self.bc)
        newstate = fgo*c_prev + igo*candidate
        ogo = self.sigmoid(np.matmul(xh, self.Wo) + self.bo)
        h_next = ogo * np.tanh(newstate)
        y = np.matmul(h_next, self.Wy) + self.by
        y = self.softmax(y)

        return h_next, newstate, y
