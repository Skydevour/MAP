import os
import math
from keras.layers.convolutional import (
    Conv3D,
    MaxPooling3D,
    AveragePooling3D
)
from keras.layers import GlobalAveragePooling3D, LeakyReLU
from keras.layers.core import (
    Dense,
    Activation,
    Reshape,
    Lambda,
    Flatten
)
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add, Concatenate, Multiply
from keras.layers import Input
from keras.regularizers import l2
from keras.models import Model
import keras.backend as K



def _bn_relu(input):
    """Helper to build a BN -> relu block (by @raghakot)."""
    norm = BatchNormalization(axis=-1)(input)
    return Activation("relu")(norm)


def _bn_relu_conv3d(**conv_params):
    """Helper to build a  BN -> relu -> conv3d block."""
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer",
                                                "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",
                                                l2(1e-4))
    use_bias = conv_params.setdefault("use_bias", False)

    def f(input):
        activation = _bn_relu(input)
        return Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, kernel_initializer=kernel_initializer,
                      padding=padding, use_bias=use_bias,
                      kernel_regularizer=kernel_regularizer)(activation)
    return f


def _conv3d_bn_relu(**conv_params):
    """Helper to build a  BN -> relu -> conv3d block."""
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer",
                                                "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",
                                                l2(1e-4))
    use_bias = conv_params.setdefault("use_bias", False)

    def f(input):
        x = Conv3D(filters=filters, kernel_size=kernel_size,
                   strides=strides, kernel_initializer=kernel_initializer,
                   padding=padding, use_bias=use_bias,
                   kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(x)
    return f


def residual_block(filters, kz=(3, 3, 3), strides=(1, 1, 1)):
    def f(input):
        x = _conv3d_bn_relu(filters=filters, kernel_size=kz)(input)
        x = Conv3D(filters, kz, strides=strides, padding="same",
                   kernel_initializer="he_normal", use_bias=False,
                   kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization(axis=-1)(x)
        identity = Conv3D(K.int_shape(x)[-1], (1, 1, 1), strides=(1, 1, 1),
                          padding="same", kernel_initializer="he_normal",
                          use_bias=False, kernel_regularizer=l2(1e-4))(input)
        x = Add()([identity, x])
        x = Activation("relu")(x)
        return x
    return f


def attention_block(filters, kz=(3, 3, 3)):
    def f(input):
        x = Conv3D(filters, kz, strides=(1, 1, 1), padding="same",
                   kernel_initializer="he_normal", use_bias=False,
                   kernel_regularizer=l2(1e-4))(input)
        x = Activation("relu")(x)
        x = Multiply()([input, x])
        return x
    return f


def resnet18(input_shape, classes):
    nb_layer = [2, 2, 2, 2]
    nb_filter = [32, 64, 128, 256]

    inpt = Input(input_shape, name="input")
    x = _conv3d_bn_relu(filters=32, kernel_size=(3, 3, 3))(inpt)
    for block_idx in range(len(nb_layer)):
        x = AveragePooling3D((2, 2, 2))(x)
        for layer_idx in range(nb_layer[block_idx]):
            x = residual_block(nb_filter[block_idx])(x)
    x = GlobalAveragePooling3D()(x)
    x = Dense(classes, activation="softmax", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(x)
    model = Model(inpt, x)
    return model


if __name__ == "__main__":
    model = att_resnet18((91, 109, 91, 1), 2)
    model.summary()