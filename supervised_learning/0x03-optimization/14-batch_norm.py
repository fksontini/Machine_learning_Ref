#!/usr/bin/env python3
""" Normalization function """

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """  creates the training operation for a neural network   """
    layer = tf.layers.Dense(
        units=n,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            mode="FAN_AVG"),
        name="layer")
    mean, var = tf.nn.moments(layer(prev), axes=[0])
    gamma = tf.ones([n])
    beta = tf.zeros([n])
    normalized = tf.nn.batch_normalization(
        layer(prev), mean, var, beta, gamma, 1e-8)
    return activation(normalized)
