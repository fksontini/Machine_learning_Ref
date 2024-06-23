#!/usr/bin/env python3
""" Transformer Applications """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Load and prep a dataset for machine translation"""

    def __init__(self):
        """initialization"""
        self.data_train, self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split=['train', 'validation'], as_supervised=True)
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """function that creates sub-word tokenizers for our dataset"""
        subtok = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus
        en = subtok(
            [en.numpy() for _, en in data], target_vocab_size=2**15)
        pt = subtok(
            [pt.numpy() for pt, _ in data], target_vocab_size=2**15)
        return pt, en
