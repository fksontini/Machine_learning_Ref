#!/usr/bin/env python3
""" Dense Block """
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """ Transition layer """
    compression_factor = int(nb_filters * compression)
    x = K.layers.BatchNormalization()(X)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(compression_factor, (1, 1),
                        kernel_initializer='he_normal', padding='same')(x)
    x = K.layers.AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x, compression_factor
