#!/usr/bin/env python3
""" Convolutional Neural Networks """

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """perform back propagation over a convolutional layer"""
    m, h, w, cprev = A_prev.shape
    m, hnew, wnew, cnew = dZ.shape
    kh, kw, cprev, cnew = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))

    oh = int((h-kh+2*ph)/sh + 1)
    ow = int((w-kw+2*pw)/sw + 1)

    prev_pad = np.pad(A_prev, pad_width=((0,), (ph,), (pw,), (0,)),
                      mode="constant", constant_values=0)

    dW = np.zeros_like(W)
    dA = np.zeros_like(prev_pad)
    db = np.zeros_like(b)

    for i in range(m):
        for x in range(oh):
            for y in range(ow):
                for c in range(cnew):
                    A = prev_pad[i, x*sh:kh+x*sh, y*sw:kw+y*sw, :]
                    dAnew = dA[i, x*sh:kh+x*sh, y*sw:kw+y*sw, :]
                    dAnew += dZ[i, x, y, c]*W[:, :, :, c]
                    dW[:, :, :, c] += A * dZ[i, x, y, c]
                    db[:, :, :, c] += dZ[i, x, y, c]

    dA = dA[:, ph:h + ph, pw:w + pw, :]
    return dA, dW, db
