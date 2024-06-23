#!/usr/bin/env python3
""" Transformer Applications """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Load and prep a dataset for machine translation"""

    def __init__(self, batch_size, max_len):
        """initialization"""
        (data_train, data_valid), metadata = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'],
            as_supervised=True, with_info=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            data_train)
        data_train = data_train.map(self.tf_encode)
        buffer_size = metadata.splits['train'].num_examples

        def filter_max_length(x, y, max_length=max_len):
            """ filtering out sentences with length > max_length"""
            return tf.logical_and(
                tf.size(x) <= max_length, tf.size(y) <= max_length)

        data_train = data_train.filter(filter_max_length)
        data_train = data_train.cache()
        data_train = data_train.shuffle(buffer_size).padded_batch(
            batch_size, padded_shapes=([None], [None]))
        self.data_train = data_train.prefetch(tf.data.experimental.AUTOTUNE)

        data_valid = data_valid.map(self.tf_encode)
        data_valid = data_valid.filter(filter_max_length)
        self.data_valid = data_valid.padded_batch(
            batch_size, padded_shapes=([None], [None]))

    def tokenize_dataset(self, data):
        """ Create sub-word tokenizers for our dataset"""
        subtok = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus
        en = subtok(
            [en.numpy() for _, en in data], target_vocab_size=2**15)
        pt = subtok(
            [pt.numpy() for pt, _ in data], target_vocab_size=2**15)
        return pt, en

    def encode(self, pt, en):
        """ Encode a translation into tokens """

        pt = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy().decode('utf-8')) + [self.tokenizer_pt.vocab_size + 1]
        en = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy().decode('utf-8')) + [self.tokenizer_en.vocab_size + 1]
        return pt, en

    def tf_encode(self, pt, en):
        """ Tensorflow wrapper for the encode instance method """
        tf_pt, tf_en = tf.py_function(
            self.encode, inp=[
                pt, en], Tout=[
                tf.int64, tf.int64])
        tf_pt.set_shape([None])
        tf_en.set_shape([None])
        return tf_pt, tf_en
