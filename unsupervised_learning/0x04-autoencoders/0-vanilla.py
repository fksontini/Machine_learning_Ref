#!/usr/bin/env python3
""" Vanilla Auto Encoder """

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """ Create an autoencoder """
    encoder_input = keras.Input(shape=(input_dims,))
    encoded = keras.layers.Dense(
        hidden_layers[0],
        activation='relu')(encoder_input)
    for i in hidden_layers[1::]:
        encoded = keras.layers.Dense(i, activation='relu')(encoded)
    latent = keras.layers.Dense(latent_dims, activation='relu')(encoded)
    encoder = keras.Model(encoder_input, latent)
    decoder_input = keras.Input(shape=(latent_dims,))
    decoded = keras.layers.Dense(
        hidden_layers[-1], activation='relu')(decoder_input)
    for i in hidden_layers[-2::-1]:
        decoded = keras.layers.Dense(i, activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(decoder_input, decoded)
    auto = keras.Model(encoder_input, decoder(encoder(encoder_input)))
    auto.compile(loss='binary_crossentropy', optimizer='adam')
    return encoder, decoder, auto
