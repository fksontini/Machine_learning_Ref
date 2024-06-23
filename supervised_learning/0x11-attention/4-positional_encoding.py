#!/usr/bin/env python3
""" Attention """
import numpy as np


def positional_encoding(max_seq_len, dm):
    """ Calculate the positional encoding for a transformer"""
    positional_embeddings = np.zeros((max_seq_len, dm))
    for position in range(max_seq_len):
        for i in range(0, dm, 2):
            positional_embeddings[position, i] = np.sin(
                position / np.power(10000, (2 * i // 2) / np.float(dm)))

            positional_embeddings[position, i + 1] = np.cos(
                position / np.power(10000, (2 * i // 2) / np.float(dm)))
    return positional_embeddings
