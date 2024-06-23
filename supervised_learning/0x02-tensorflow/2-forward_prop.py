#!/usr/bin/env python3
""" Forward propagation """

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """ creates the forward propagation graph for the neural network """
    ln = x
    for i in range(len(layer_sizes)):
        ln = create_layer(ln, layer_sizes[i],
                          activation=activations[i])
    return ln
