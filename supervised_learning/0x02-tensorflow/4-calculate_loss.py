#!/usr/bin/env python3
""" Calculate loss """

import tensorflow as tf


def calculate_loss(y, y_pred):
    """ calculates the softmax cross-entropy loss of a prediction """
    return tf.losses.softmax_cross_entropy(
        y,
        y_pred,
        weights=1.0,
        label_smoothing=0,
        scope=None,
        loss_collection=tf.GraphKeys.LOSSES,
    )
