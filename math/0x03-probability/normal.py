#!/usr/bin/env python3
""" Create a class Normal """


class Normal:
    """  represent an Normal distribution   """

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """ Class contructor """
        self.data = data
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)

        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            a = 0
            for i in data:
                a += (i - self.mean) ** 2
            self.stddev = (a / len(data)) ** (1/2)

    def z_score(self, x):
        """ Calculate the z-score of a given x-value """

        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ Calculate the x-value of a given z-score """

        return z * self.stddev + self.mean

    def pdf(self, x):
        """ Calculate the value of the PDF for a given x-value """

        p1 = self.stddev * ((2 * self.pi) ** 0.5)
        p2 = self.e ** (- 0.5 * (((x - self.mean) / self.stddev) ** 2))
        return p2 / p1

    def erf(self, x):
        """ Error function """

        devl = (x - x ** 3 / 3 + x ** 5 / 10 - x ** 7 / 42 + x ** 9 / 216)
        return (2 / self.pi ** 0.5) * devl

    def cdf(self, x):
        """ Calculate the value of the CDF for a given x-value """

        a = 1 + self.erf((x - self.mean) / (self.stddev * (2 ** 0.5)))
        return a / 2
