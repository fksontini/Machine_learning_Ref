#!/usr/bin/env python3
""" Dense Block """
import tensorflow.keras as K


def H(x, num_filters):
    """ conv layers """
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(
        4 * num_filters,
        (1,
         1),
        kernel_initializer='he_normal',
        padding='same')(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(
        num_filters,
        (3,
         3),
        kernel_initializer='he_normal',
        padding='same')(x)
    return x


def dense_block(X, nb_filters, growth_rate, layers):
    """Build a dense block """
    for i in range(layers):
        layer = H(X, growth_rate)
        X = K.layers.Concatenate()([X, layer])
        nb_filters += growth_rate
    return X, nb_filters
