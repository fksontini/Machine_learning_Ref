#!/usr/bin/env python3
""" Keras """
import tensorflow.keras as K


def save_config(network, filename):
    """save a model’s configuration in JSON format"""
    with open(filename, "w") as f:
        f.write(network.to_json())
    return None


def load_config(filename):
    """load a model with a specific configuration"""
    with open(filename, "r") as f:
        reading = f.read()
    modelload = K.models.model_from_json(reading)
    return modelload
