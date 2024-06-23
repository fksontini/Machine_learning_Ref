#!/usr/bin/env python3
""" Identity Block """
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """ Build a projection block """
    initializer = K.initializers.he_normal()
    layer = K.layers.Conv2D(
        filters[0],
        (1,
         1), strides=s,
        padding='same',
        kernel_initializer=initializer)(A_prev)
    layer = K.layers.BatchNormalization()(layer)
    layer = K.layers.Activation('relu')(layer)
    layer = K.layers.Conv2D(
        filters[1],
        (3,
         3),
        padding='same',
        kernel_initializer=initializer)(layer)
    layer = K.layers.BatchNormalization()(layer)
    layer = K.layers.Activation('relu')(layer)
    layer = K.layers.Conv2D(
        filters[2],
        (1,
         1),
        padding='same',
        kernel_initializer=initializer)(layer)
    layer = K.layers.BatchNormalization()(layer)
    A_prev = K.layers.Conv2D(
        filters[2],
        (1,
         1), strides=s,
        padding='same',
        kernel_initializer=initializer)(A_prev)
    A_prev = K.layers.BatchNormalization()(A_prev)
    output = K.layers.Add()([layer, A_prev])
    output = K.layers.Activation('relu')(output)
    return output
