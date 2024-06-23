#!/usr/bin/env python3
"""  Convolution and pooling"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """performs a convolution on grajscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    if padding == 'valid':
        pad_h = 0
        pad_w = 0
    elif padding == 'same':
        pad_h = int((((h - 1) * sh + kh - h) / 2)) + 1
        pad_w = int((((w - 1) * sw + kw - w) / 2)) + 1
        output = np.zeros((m, h, w))
    else:
        pad_h = padding[0]
        pad_w = padding[1]

    output_h = int(((h - kh + 2 * pad_h) / sh) + 1)
    output_w = int(((w - kw + 2 * pad_w) / sw) + 1)

    output = np.zeros((m, output_h, output_w))
    img_pad = np.pad(
        array=images,
        pad_width=((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=0)

    for x in range(output_h):
        for y in range(output_w):
            output[:, x, y] = (img_pad[:, x*sh:kh + x * sh,
                                       y*sw:kw + y*sw]*kernel).sum(axis=(1, 2))
    return output
