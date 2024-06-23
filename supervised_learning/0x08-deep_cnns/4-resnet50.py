#!/usr/bin/env python3
""" Resnet-50 """
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ Build the ResNet-50 architecture """
    input_layer = K.Input(shape=(224, 224, 3))
    initializer = K.initializers.he_normal()

    # Stage 1
    layer = K.layers.Conv2D(64, (7, 7), padding='same', strides=2,
                            kernel_initializer=initializer)(input_layer)
    layer = K.layers.BatchNormalization()(layer)
    layer = K.layers.Activation('relu')(layer)
    layer = K.layers.MaxPooling2D(
        (3, 3), strides=(
            2, 2), padding='same')(layer)

    # stage2
    layer = projection_block(layer, [64, 64, 256], s=1)
    layer = identity_block(layer, [64, 64, 256])
    layer = identity_block(layer, [64, 64, 256])

    # stage3
    layer = projection_block(layer, [128, 128, 512], s=2)
    layer = identity_block(layer, [128, 128, 512])
    layer = identity_block(layer, [128, 128, 512])
    layer = identity_block(layer, [128, 128, 512])

    # stage4
    layer = projection_block(layer, [256, 256, 1024], s=2)
    layer = identity_block(layer, [256, 256, 1024])
    layer = identity_block(layer, [256, 256, 1024])
    layer = identity_block(layer, [256, 256, 1024])
    layer = identity_block(layer, [256, 256, 1024])
    layer = identity_block(layer, [256, 256, 1024])

    # stage5
    layer = projection_block(layer, [512, 512, 2048], s=2)
    layer = identity_block(layer, [512, 512, 2048])
    layer = identity_block(layer, [512, 512, 2048])

    # AveragePooling
    layer = K.layers.AveragePooling2D((7, 7))(layer)

    # softmax
    output = K.layers.Dense(
        1000,
        activation='softmax',
        kernel_initializer=initializer)(layer)

    model = K.models.Model(inputs=input_layer, outputs=output)
    return model
