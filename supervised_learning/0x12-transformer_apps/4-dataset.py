#!/usr/bin/env python3
""" Transformer Applications """
import tensorflow.compat.v2 as tf


def padding_mask(input):
    """ Create a padding mask"""
    return tf.cast(tf.math.equal(input, 0), tf.float32)[
        :, tf.newaxis, tf.newaxis, :]


def create_masks(inputs, target):
    """creates all masks for training/validation"""
    batch_size, seq_len_out = target.shape
    batch_size, _ = inputs.shape
    encoder_mask = padding_mask(inputs)
    decoder_mask = padding_mask(inputs)
    look_ahead_mask = 1 - \
        tf.linalg.band_part(tf.ones(shape=(batch_size,
                            1, seq_len_out, seq_len_out)), -1, 0)
    decoder_target_padding_mask = padding_mask(target)
    combined_mask = tf.maximum(decoder_target_padding_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
