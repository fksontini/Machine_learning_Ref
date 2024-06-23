#!/usr/bin/env python3
""" DeepNeuralNetwork Class """


import numpy as np
import matplotlib.pyplot as plt
import pickle


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
        self.__layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(len(layers)):
            if i == 0:
                self.__weights["W" + str(i + 1)] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)

                self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))
            else:
                self.__weights["W" + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])

                self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def layers(self):
        return self.__layers

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache["A0"] = X
        for i in range(self.__L):
            z = np.dot(self.__weights["W" + str(i + 1)], self.__cache[
                "A" + str(i)]) + (self.__weights["b" + str(i + 1)])
            self.__cache["A" + str(i + 1)] = 1 / (1 + np.exp(-z))
        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """ Calculate the cost of the model using logistic regression """
        m = Y.shape[1]
        T = np.transpose(A)
        L = np.transpose(np.log(1.0000001 - A))
        cst = np.squeeze(-1 / m * (np.dot(Y, np.log(T)) + np.dot(1 - Y, L)))
        return cst

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        self.forward_prop(X)
        P = np.where(self.__cache["A" + str(self.__L)] < 0.5, 0, 1)
        c = self.cost(Y, self.__cache["A" + str(self.__L)])
        return P, c

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates one pass of gradient descent on the neural network """
        m = Y.shape[1]
        dz = cache["A" + str(self.__L)] - Y
        for i in range(self.L, 0, -1):
            A = cache["A" + str(i - 1)]
            dW = np.matmul(dz, A.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            dz = np.matmul(self.__weights[
                "W" + str(i)].T, dz) * (A * (1 - A))
            self.__weights["W" + str(i)] -= alpha * dW
            self.__weights["b" + str(i)] -= alpha * db

    def train(
        self, X, Y,
        iterations=5000, alpha=0.05,
        verbose=True, graph=True, step=100
    ):
        """ Trains the neural network """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        axey = []
        axex = []
        for iteration in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            if verbose and iteration % 100 == 0:
                print("Cost after {} iterations: {}".format(
                    iteration, self.cost(Y, A)))
                axey.append(self.cost(Y, A))
                axex.append(iteration)
        if graph:
            plt.plot(axex, axey, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """ Saves the instance object to a file in a pickle format """
        if not filename.endswith(".pkl"):
            filename += ".pkl"
            with open(filename, "wb") as f:
                pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """loads pickled instance object"""
        try:
            with open(filename, "rb") as f:
                ld = pickle.load(f)
            return ld
        except Exception:
            return None
