#!/usr/bin/env python3
""" Create layers """

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ calculates the accuracy of a prediction """
    return tf.math.reduce_mean(
        tf.cast(
            tf.math.equal(
                tf.math.argmax(
                    y,
                    axis=-1),
                tf.math.argmax(
                    y_pred,
                    axis=-1)),
            tf.float32))
