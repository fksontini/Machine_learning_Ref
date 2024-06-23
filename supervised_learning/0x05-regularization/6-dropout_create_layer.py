#!/usr/bin/env python3
""" Regularization """

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Creates a layer of a neural network using dropout"""
    kernel_init = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    drop = tf.layers.Dropout(keep_prob)
    lay_er = tf.layers.Dense(units=n, activation=activation,
                             kernel_initializer=kernel_init,
                             )
    return drop(lay_er(prev))
