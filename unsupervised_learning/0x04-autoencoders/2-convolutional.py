#!/usr/bin/env python3
""" Convolutional Autoencoder """

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """ Creates  convolutional autoencoder """
    encoder_input = keras.Input(shape=input_dims)
    encoded = keras.layers.Conv2D(
        filters[0], (3, 3), activation='relu', padding='same')(encoder_input)
    encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
    for i in range(1, len(filters)):
        encoded = keras.layers.Conv2D(
            filters[i], (3, 3), activation='relu', padding='same')(encoded)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
    encoder = keras.Model(encoder_input, encoded)
    decoder_input = keras.Input(shape=latent_dims)
    decoded = keras.layers.Conv2D(
        filters[-1], (3, 3), activation='relu', padding='same')(decoder_input)
    decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    for i in range(len(filters) - 2, 0, -1):
        decoded = keras.layers.Conv2D(
            filters[i], (3, 3), padding='same', activation='relu')(decoded)
        decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    decoded = keras.layers.Conv2D(
        filters[0], (3, 3), padding='valid', activation='relu')(decoded)
    decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    decoded = keras.layers.Conv2D(
        input_dims[-1], (3, 3), activation='sigmoid', padding='same')(decoded)
    decoder = keras.Model(decoder_input, decoded)
    auto = keras.Model(encoder_input, decoder(encoder(encoder_input)))
    auto.compile(loss='binary_crossentropy', optimizer='adam')

    return encoder, decoder, auto
