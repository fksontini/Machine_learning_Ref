#!/usr/bin/env python3
""" DeepNeuralNetwork Class """


import numpy as np


class DeepNeuralNetwork:
    """ Define a Deep Neural Network performing binary classification """

    def __init__(self, nx, layers):
        """ Instantination """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or layers == []:
            raise TypeError("layers must be a list of positive integers")
        if not all(
                map(lambda x: x > 0 and isinstance(x, int), layers)):
            raise TypeError("layers must be a list of positive integers")
        self.layers = layers
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(len(layers)):
            if i == 0:
                self.weights["W" + str(i + 1)] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)

                self.weights["b" + str(i + 1)] = np.zeros((layers[i], 1))
            else:
                self.weights["W" + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])

                self.weights["b" + str(i + 1)] = np.zeros((layers[i], 1))
