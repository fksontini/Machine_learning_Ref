#!/usr/bin/env python3
""" Convolutional Neural Networks """

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ perform a forward propagation over a pooling layer """
    m, h, w, cprev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    oh = int((h-kh)/sh + 1)
    ow = int((w-kh)/sw + 1)
    z = np.empty((m, oh, ow, cprev))

    for x in range(oh):
        for y in range(ow):
            if mode == 'max':
                z[:, x, y, :] = np.max(
                    A_prev[:, x*sh:kh+x*sh, y*sw:kw+y*sw, :],
                    axis=(1, 2))
            if mode == 'avg':
                z[:, x, y, :] = np.average(
                    A_prev[:, x*sh:kh+x*sh, y*sw:kw+y*sw, :],
                    axis=(1, 2))
    return z
