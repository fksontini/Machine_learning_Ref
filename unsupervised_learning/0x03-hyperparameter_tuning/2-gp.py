#!/usr/bin/env python3
""" Gaussian Process """

import numpy as np


class GaussianProcess:
    """ Represent a noiseless 1D Gaussian process """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """ Class constructor """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """ Calculate the covariance kernel matrix between two matrices """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
            np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """ Predict the mean and standard deviation
        of points in a Gaussian process """
        s = X_s.shape[0]
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)
        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(s,)
        sigma = (K_ss - K_s.T.dot(K_inv).dot(K_s)).diagonal()
        return mu, sigma

    def update(self, X_new, Y_new):
        """ Update a Gaussian Process """
        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(self.X, self.X)
