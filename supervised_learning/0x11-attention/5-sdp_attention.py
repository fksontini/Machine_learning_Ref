#!/usr/bin/env python3
""" Attention """
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """ Calculate the scaled dot product attention """

    matmul = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scale = matmul / tf.math.sqrt(dk)
    if mask is not None:
        scale += (mask * -1e9)
    softmax = tf.nn.softmax(scale, axis=-1)
    output = tf.matmul(softmax, V)
    return output, softmax
