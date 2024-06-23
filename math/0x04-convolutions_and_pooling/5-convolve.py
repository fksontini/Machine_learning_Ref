#!/usr/bin/env python3
"""  Convolution and pooling"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """performs a convolution on images using multiple kernel"""
    m, h, w, c = images.shape
    kh, kw, c, nc = kernels.shape
    sh, sw = stride

    if padding == 'valid':
        output_h = int(np.floor(((h - kh)) / sh + 1))
        output_w = int(np.floor(((w - kw)) / sw + 1))
        output = np.zeros((m, output_h, output_w, nc))
        img_pad = images.copy()
    if padding == "same":
        pad_h = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pad_w = int(np.ceil(((w - 1) * sw + kw - w) / 2))
        output_h = int(np.floor((h - kh + 2 * pad_h) / sh) + 1)
        output_w = int(np.floor((w - kw + 2 * pad_w) / sw) + 1)

        output = np.zeros((m, output_h, output_w, nc))
        img_pad = np.pad(
            array=images,
            pad_width=((0,), (pad_h,), (pad_w,), (0,)),
            mode="constant",
            constant_values=0)
    if isinstance(padding, tuple):
        pad_h, pad_w = padding
        output_h = int(np.floor((h - kh + 2 * pad_h) / sh) + 1)
        output_w = int(np.floor((w - kw + 2 * pad_w) / sw) + 1)

        output = np.zeros((m, output_h, output_w, nc))
        img_pad = np.pad(
            array=images,
            pad_width=((0,), (pad_h,), (pad_w,), (0,)),
            mode="constant",
            constant_values=0)
    for x in range(output_h):
        for y in range(output_w):
            for c in range(nc):
                output[:, x, y, c] = (
                    img_pad[:, x*sh:kh+x*sh, y*sw:kw+y*sw, :]*kernels[
                        :, :, :, c]).sum(axis=(1, 2, 3))
    return output
