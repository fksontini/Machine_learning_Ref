#!/usr/bin/env python3
""" Keras """
import tensorflow.keras as K


def save_model(network, filename):
    """save an entire model"""
    network.save(filename)
    return None


def load_model(filename):
    """load an entire model"""
    return K.models.load_model(filename)
