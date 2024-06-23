#!/usr/bin/env python3
""" Convolutional Neural Networks """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='avg'):
    """perform back propagation over a pooling layer"""
    m, h, w, cnew = A_prev.shape
    m, hnew, wnew, cnew = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for x in range(hnew):
            for y in range(wnew):
                for c in range(cnew):
                    A = A_prev[i, x*sh:kh+x*sh, y*sw:kw+y*sw, c]
                    dAu = dA_prev[i, x*sh:kh+x*sh, y*sw:kw+y*sw, c]

                    if mode == 'max':
                        mask = (A == np.max(A))
                        dAu += np.multiply(dA[i, x, y, c], mask)

                    if mode == 'avg':
                        dAu += dA[i, x, y, c]/(kh*kw)
    return dA_prev
