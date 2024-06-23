#!/usr/bin/env python3
"""Poly Derivate"""


def poly_derivative(poly):
    """   Calculate the derivative of a polynomial   """
    lp = []
    if poly == [] or not isinstance(poly, list):
        return None
    if len(poly) == 1:
        return [0]
    for i in range(1, len(poly)):
        lp.append(i * poly[i])
    return lp
