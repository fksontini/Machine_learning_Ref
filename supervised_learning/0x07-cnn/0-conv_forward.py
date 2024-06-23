#!/usr/bin/env python3
""" Convolutional Neural Networks """

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ perform forward propagation over a convolutional layer """
    m, h, w, cprev = A_prev.shape
    kh, kw, cpev, cnew = W.shape
    sh, sw = stride
    if padding == 'valid':
        ph = 0
        pw = 0
    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))

    oh = int((h-kh+2*ph)/sh + 1)
    ow = int((w-kw+2*pw)/sw + 1)
    z = np.zeros((m, oh, ow, cnew))
    prev_pad = np.pad(A_prev, pad_width=((0,), (ph,), (pw,), (0,)),
                      mode="constant", constant_values=0)
    for i in range(oh):
        for j in range(ow):
            for c in range(cnew):
                piece = prev_pad[:, i*sh:kh+i*sh, j*sw:kw+j*sw, :]
                z[:, i, j, c] = activation(np.sum(
                    piece*W[:, :, :, c], axis=(1, 2, 3)) + b[:, :, :, c])
    return z
