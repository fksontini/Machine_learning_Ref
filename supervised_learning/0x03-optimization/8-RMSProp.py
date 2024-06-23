#!/usr/bin/env python3
""" Normalization function """

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ creates the training operation for a neural network """
    return tf.train.RMSPropOptimizer(
        alpha, beta2, epsilon=epsilon).minimize(loss)
