from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)
import six
from math import ceil
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Concatenate,
    Add
)
from keras.layers.convolutional import (
    Conv3D,
    AveragePooling3D,
    MaxPooling3D
)
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K


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


def conv_block(x, growth_rate, channel_axis=-1):
    x1 = _bn_relu_conv3d(filters=growth_rate, kernel_size=(3, 3, 3))(x)
    x = Concatenate(axis=channel_axis)([x, x1])
    return x


def transition(x, reduction=0.5):
    """Apply BatchNorm, Relu 1x1x1Conv3D, optional dropout and Maxpooling3D
        :param x: keras model
        :param concat_axis: int -- index of contatenate axis
        :param nb_filter: int -- number of filters
        :param dropout_rate: int -- dropout rate
        :param weight_decay: int -- weight decay factor

        :returns: model
        :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool
    """
    nb_filter = K.int_shape(x)[-1]
    x = _bn_relu_conv3d(filters=int(nb_filter*reduction), kernel_size=(1, 1, 1))(x)
    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)
    return x


def dense_block(inpt, nb_layer, growth_rate=12, channel_axis=-1):
    x = inpt
    for i in range(nb_layer):
        x = conv_block(x, growth_rate, channel_axis)  # (bz, h/2, w/2, d/2, 12)
    return x



def denseNet(input_shape, nb_class, nb_dense_block=4):

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = 64
    nb_layer = [6, 12, 24, 6] #[6,8,12,6]
    inpt = Input(shape=input_shape)  # (bz, h, w, d, c)
    #x0 = _conv_bn_relu3D(filters=filters, kernel_size=(3, 3, 3),  # (bz, h/2, w/2, d/2, 64)
    #                     strides=(2, 2, 2))(inpt)
    x = Conv3D(filters, kernel_size=(7, 7, 7), strides=(2, 2, 2), use_bias=False,# (bz, h/2, w/2, d/2, 64)
               padding="same", kernel_initializer="he_normal",
               kernel_regularizer=l2(1e-4))(inpt)
    x = _bn_relu(x)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding="same")(x)

    #rdb_list = []
    for block_idx in range(nb_dense_block-1):
        x = dense_block(x, nb_layer[block_idx], growth_rate=12, channel_axis=channel_axis)
        x = transition(x)

    x = dense_block(x, nb_layer[nb_dense_block-1], growth_rate=12, channel_axis=channel_axis)
    #x = Concatenate(axis=channel_axis)(rdb_list)
    #if globalSkip:
    #    x0 = _bn_relu_conv3d(filters=K.int_shape(x)[channel_axis], kernel_size=(1, 1, 1))(x0)
    #    x = Add()([x0, x])

    x = _bn_relu(x)
    x = AveragePooling3D(K.int_shape(x)[1:-1])(x)
    x = Flatten()(x)
    pred = Dense(nb_class, activation="sigmoid", kernel_initializer="he_normal",
                 kernel_regularizer=l2(1e-4))(x)

    model = Model(inpt, pred)
    return model



def denseNet40(input_shape, nb_class, nb_dense_block=3):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = 16
    nb_layer = [12, 12, 12]  # [6,8,12,6]
    inpt = Input(shape=input_shape)  # (bz, h, w, d, c)
    # x0 = _conv_bn_relu3D(filters=filters, kernel_size=(3, 3, 3),  # (bz, h/2, w/2, d/2, 64)
    #                     strides=(2, 2, 2))(inpt)
    x = Conv3D(filters, kernel_size=(3, 3, 3), strides=(2, 2, 2), use_bias=False,  # (bz, h/2, w/2, d/2, 64)
               padding="same", kernel_initializer="he_normal",
               kernel_regularizer=l2(1e-4))(inpt)

    x = _bn_relu(x)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding="same")(x)

    for block_idx in range(nb_dense_block - 1):
        x = dense_block(x, nb_layer[block_idx], growth_rate=12, channel_axis=channel_axis)
        x = transition(x)

    x = dense_block(x, nb_layer[nb_dense_block - 1], growth_rate=12, channel_axis=channel_axis)

    x = _bn_relu(x)
    x = AveragePooling3D(K.int_shape(x)[1:-1])(x)
    x = Flatten()(x)
    pred = Dense(nb_class, activation="sigmoid", kernel_initializer="he_normal",
                 kernel_regularizer=l2(1e-4))(x)

    model = Model(inpt, pred)
    return model



if __name__ == '__main__':
    model = RDBNet((224, 224, 95, 1), 3)
    model.summary()