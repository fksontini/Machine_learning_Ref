#!/usr/bin/env python3
""" Create layers """

import tensorflow as tf


def create_layer(prev, n, activation):
    """ Create layer """
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            mode="FAN_AVG"),
        name="layer",
    )
    return layer(prev)
