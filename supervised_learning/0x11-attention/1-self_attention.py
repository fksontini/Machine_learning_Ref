#!/usr/bin/env python3
""" Attention """
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """ Calculate the attention for machine translation """
    def __init__(self, units):
        """ initialization"""
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """  Bahdanau implementation """
        c = tf.expand_dims(s_prev, 1)
        m = self.V(tf.nn.tanh(self.W(c) + self.U(hidden_states)))
        s = tf.nn.softmax(m, axis=1)
        z = tf.reduce_sum(s * hidden_states, axis=1)
        return z, s
