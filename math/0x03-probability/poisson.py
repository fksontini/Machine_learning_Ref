#!/usr/bin/env python3
""" Create a class Poisson """


class Poisson:
    """  represent a poisson distribution   """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """ Class contructor """
        self.data = data
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """ Calculate the value of the PMF for a given number of successes """

        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        fact = 1
        for i in range(2, k + 1):
            fact = fact * i
        pmf = (self.lambtha ** k) * (self.e ** (- self.lambtha)) / fact
        return pmf

    def cdf(self, k):
        """ Calculate the value of the CDF for a given number of successes """

        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        cumul = 0
        for i in range(k + 1):
            cumul += self.pmf(i)
        return cumul
