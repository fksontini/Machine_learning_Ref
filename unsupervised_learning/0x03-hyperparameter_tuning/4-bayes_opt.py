#!/usr/bin/env python3
""" Bayesian Optimization """

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """ Perform Bayesian optimization on a noiseless 1D Gaussian process """

    def __init__(
            self,
            f,
            X_init,
            Y_init,
            bounds,
            ac_samples,
            l=1,
            sigma_f=1,
            xsi=0.01,
            minimize=True):
        """ Initialization """
        low, high = bounds
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(low, high, ac_samples).reshape((-1, 1))
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """ Calculate the next best sample location """
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize is True:
            fpredict = np.min(self.gp.Y)
        else:
            fpredict = np.max(self.gp.Y)
        Z = (fpredict - mu - self.xsi) / sigma
        EI = (fpredict - mu - self.xsi) * \
            norm.cdf(Z) + sigma * norm.pdf(Z)
        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI
