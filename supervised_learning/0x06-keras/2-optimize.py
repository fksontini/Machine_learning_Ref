#!/usr/bin/env python3
""" Keras """

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """ sets up Adam optimization for a keras model """
    opt = K.optimizers.Adam(alpha, beta1, beta2)
    network.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
