#!/usr/bin/env python3
"""  Convolution and pooling"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ perform a same convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h = padding[0]
    pad_w = padding[1]
    final_h = h + 2 * pad_h - kh + 1
    final_w = w + 2 * pad_w - kw + 1
    output = np.zeros((m, final_h, final_w))
    img_pad = np.pad(
        array=images,
        pad_width=((0,), (pad_h,), (pad_w,)),
        mode="constant",
        constant_values=0)

    for x in range(final_h):
        for y in range(final_w):
            output[:, x, y] = (img_pad[:, x:kh+x, y:kw+y] * kernel
                               ).sum(axis=(1, 2))
    return output
