#!/usr/bin/env python3
""" Variational Autoencoder """

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """ Create a variational autoencoder """

    encoder_input = keras.Input(shape=(input_dims,))
    encoded = keras.layers.Dense(
        hidden_layers[0],
        activation='relu')(encoder_input)
    for i in hidden_layers[1::]:
        encoded = keras.layers.Dense(i, activation='relu')(encoded)

    z_mean = keras.layers.Dense(latent_dims, activation=None)(encoded)
    z_log_sigma = keras.layers.Dense(latent_dims, activation=None)(encoded)
    epsilon = keras.backend.random_normal(
        shape=(latent_dims,), mean=0.0, stddev=1.0)
    sample = z_mean + keras.backend.exp(z_log_sigma / 2) * epsilon
    z = keras.layers.Lambda(sample, output_shape=(
        latent_dims,))([z_mean, z_log_sigma])

    encoder = keras.Model(encoder_input, [z_mean, z_log_sigma, z])
    decoder_input = keras.Input(shape=(latent_dims,))
    decoded = keras.layers.Dense(
        hidden_layers[-1], activation='relu')(decoder_input)
    for i in hidden_layers[-2::-1]:
        decoded = keras.layers.Dense(i, activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(decoder_input, decoded)
    output = decoder(encoder(encoder_input)[2])
    auto = keras.Model(encoder_input, output)

    reconstruction_loss = keras.losses.binary_crossentropy(
        encoder_input, output)
    reconstruction_loss *= input_dims
    kl_loss = 1 + z_log_sigma - keras.backend.square(z_mean) \
        - keras.backend.exp(z_log_sigma)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto
