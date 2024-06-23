#!/usr/bin/env python3
"""  Convolution and pooling"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """ perform a same convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    if kh % 2 == 1:
        pad_h = (kh - 1) // 2
    else:
        pad_h = kh // 2
    if kw % 2 == 1:
        pad_w = (kw - 1) // 2
    else:
        pad_w = kw // 2
    output = np.zeros((m, h, w))
    img_pad = np.pad(
        array=images,
        pad_width=((0,), (pad_h,), (pad_w,)),
        mode="constant",
        constant_values=0)

    for x in range(h):
        for y in range(w):
            output[:, x, y] = (img_pad[:, x:kh+x, y:kw+y] * kernel
                               ).sum(axis=(1, 2))
    return output
