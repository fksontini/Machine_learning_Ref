#!/usr/bin/env python3
""" Convolutional Neural Networks """

import tensorflow.keras as K


def lenet5(X):
    """build a modified version of the LeNet-5 architecture"""
    init = K.initializers.he_normal()
    conv1 = K.layers.Conv2D(6, (5, 5),
                            padding='same', kernel_initializer=init,
                            activation='relu')(X)
    Maxpool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    conv2 = K.layers.Conv2D(16, (5, 5),
                            padding='valid', kernel_initializer=init,
                            activation='relu')(Maxpool1)
    Maxpool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    FC = K.layers.Flatten()(Maxpool2)
    FC1 = K.layers.Dense(120, kernel_initializer=init,
                         activation='relu')(FC)
    FC2 = K.layers.Dense(84, kernel_initializer=init,
                         activation='relu')(FC1)
    output = K.layers.Dense(10, kernel_initializer=init,
                            activation='softmax')(FC2)
    model = K.models.Model(inputs=X, outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])
    return model
