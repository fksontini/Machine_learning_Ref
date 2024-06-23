#!/usr/bin/env python3
""" NeuronNetwork Class """


import numpy as np


class NeuralNetwork:
    """ Define a neural network with one hidden layer """

    def __init__(self, nx, nodes):
        """Instantination"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.nx = nx
        self.nodes = nodes
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network """
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1/(1 + (np.exp(-z1)))
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1/(1 + (np.exp(-z2)))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ Calculate the cost of the model using logistic regression """
        m = Y.shape[1]
        T = np.transpose(A)
        L = np.transpose(np.log(1.0000001 - A))
        cst = np.squeeze(-1 / m * (np.dot(Y, np.log(T)) + np.dot(1 - Y, L)))
        return cst

    def evaluate(self, X, Y):
        """ Evaluate the neuron’s predictions """

        self.forward_prop(X)
        P = np.where(self.__A2 < 0.5, 0, 1)
        c = self.cost(Y, self.__A2)
        return P, c

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        dz2 = A2 - Y
        m = Y.shape[1]
        dz1 = np.matmul(self.W2.T, dz2) * (A1 * (1 - A1))
        self.__W2 = self.__W2 - alpha * np.matmul(dz2, A1.T) / m
        self.__b2 = self.__b2 - alpha * np.sum(dz2, axis=1, keepdims=True) / m
        self.__W1 = self.__W1 - alpha * np.matmul(dz1, X.T) / m
        self.__b1 = self.__b1 - alpha * np.sum(dz1, axis=1, keepdims=True) / m

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        for _ in range(iterations):
            self.__A1, self.__A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
        return self.evaluate(X, Y)
