#!/usr/bin/env python3
""" Inception Model"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ Build an inception block """
    conv1 = K.layers.Conv2D(
        filters[0], (1, 1), padding='same', activation='relu')(A_prev)
    conv1_3 = K.layers.Conv2D(
        filters[1], (1, 1), padding='same', activation='relu')(A_prev)
    conv3 = K.layers.Conv2D(
        filters[2], (3, 3), padding='same', activation='relu')(conv1_3)
    conv1_5 = K.layers.Conv2D(
        filters[3], (1, 1), padding='same', activation='relu')(A_prev)
    conv5 = K.layers.Conv2D(
        filters[4], (5, 5), padding='same', activation='relu')(conv1_5)
    pool = K.layers.MaxPooling2D(
        (3, 3), strides=(
            1, 1), padding='same')(A_prev)
    convp_1 = K.layers.Conv2D(
        filters[5], (1, 1), padding='same', activation='relu')(pool)
    output = K.layers.concatenate([conv1, conv3, conv5, convp_1], axis=-1)
    return output
