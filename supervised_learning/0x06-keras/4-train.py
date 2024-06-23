#!/usr/bin/env python3
""" Keras """
import tensorflow.keras as K


def train_model(
        network,
        data,
        labels,
        batch_size,
        epochs,
        verbose=True,
        shuffle=False):
    """ train a model using mini-batch gradient descent """
    history = network.fit(
        x=data,
        y=labels,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        shuffle=shuffle)
    return history
