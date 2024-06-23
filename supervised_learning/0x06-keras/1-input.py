#!/usr/bin/env python3
""" Keras """

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ build a neural network with the Keras library """
    input_1 = K.Input(shape=(nx,))
    x = input_1
    for i in range(len(layers)):
        if i != 0:
            x = K.layers.Dropout(1 - keep_prob)(x)
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha))(x)
    return K.Model(inputs=input_1, outputs=x)
