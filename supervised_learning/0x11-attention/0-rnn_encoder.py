#!/usr/bin/env python3
""" Attention """
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """ Encode for machine translation """
    def __init__(self, vocab, embedding, units, batch):
        """ initialization """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units, recurrent_initializer="glorot_uniform",
            return_sequences=True, return_state=True)

    def initialize_hidden_state(self):
        """Initialize the hidden states for RNN cell to a tensor of zeros"""
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """call funtion """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial)
        return outputs, hidden
