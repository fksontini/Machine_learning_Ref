#!/usr/bin/env python3
""" Attention """
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """ Perform multi head attention """
    def __init__(self, dm, h):
        """initialization"""
        super(MultiHeadAttention, self).__init__()
        self.dm = dm
        self.h = h
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """call function  """
        batch_size = tf.shape(Q)[0]

        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        q = tf.reshape(q, (batch_size, -1, self.h, self.depth))
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.reshape(k, (batch_size, -1, self.h, self.depth))
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.reshape(v, (batch_size, -1, self.h, self.depth))
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        sdpsoftmax, sdpoutput = sdp_attention(q, k, v, mask)
        sdpsoftmax = tf.transpose(sdpsoftmax, perm=[0, 2, 1, 3])

        concat_sdp = tf.reshape(sdpsoftmax, (batch_size, -1, self.dm))

        output = self.linear(concat_sdp)

        return output, sdpoutput
