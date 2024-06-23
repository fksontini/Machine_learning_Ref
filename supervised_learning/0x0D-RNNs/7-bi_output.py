#!/usr/bin/env python3
""" Recurrent Neural Network """
import numpy as np


class BidirectionalCell:
    """ Represent a bidirectional cell of an RNN """

    def __init__(self, i, h, o):
        """ initialization"""
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """Softmax Activation Function"""
        expo = np.exp(x - np.max(x))
        return expo / expo.sum(axis=1, keepdims=True)

    def sigmoid(self, x):
        """Sigmoid Activation Function"""
        return np.exp(-np.logaddexp(0, -x))

    def forward(self, h_prev, x_t):
        """Calculate the hidden state in the forward
        direction for one time step """
        xh = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(xh, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """ Calculate the hidden state in the backward
        direction for one time step"""
        xh = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(xh, self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        """ Calculate all outputs for the RNN """
        t, m, _ = H.shape
        o = self.Wy.shape[1]
        Y = np.empty(shape=(t, m, o))
        for s in range(H.shape[0]):
            Y[s, ...] = self.softmax(np.matmul(H[s, ...], self.Wy) + self.by)
        return Y
