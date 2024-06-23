#!/usr/bin/env python3
""" Normalization function """

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """  creates the training operation for a neural network   """
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)
