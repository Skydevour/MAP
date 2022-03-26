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
from keras.layers.merge import add, concatenate, multiply
from keras.layers import Input
from keras.regularizers import l2
from keras.models import Model
import keras.backend as K


def squeeze_excite_block(input_tensor, ratio=16):
    """ Create a channel-wise squeeze-excite block

    Args:
        input_tensor: input Keras tensor
        ratio: number of output filters

    Returns: a Keras tensor

    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    """
    init = input_tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]

    se = AveragePooling3D(K.int_shape(init)[1:-1])(init)
    se = Conv3D(filters//ratio, kernel_size=(1, 1, 1), kernel_initializer='he_normal', use_bias=False)(se)
    se = Activation("relu")(se)
    se = Conv3D(filters, kernel_size=(1, 1, 1), kernel_initializer='he_normal', use_bias=False)(se)
    se = Activation("sigmoid")(se)
    x = multiply([init, se])
    return x


def _bn_relu(input):
    """Helper to build a BN -> relu block (by @raghakot)."""
    norm = BatchNormalization(axis=-1)(input)
    return Activation("relu")(norm)


def _bn_relu_Conv3D(filters, kernel_size, name, flag, is_first_layer_of_first_block, strides):
    def f(input):
        if is_first_layer_of_first_block:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            x = Conv3D(filters, kernel_size, strides=strides, padding="same",
                       use_bias=False, kernel_initializer="he_normal",
                       kernel_regularizer=l2(1e-4), name=name + "conv{}".format(flag))(input)
        else:
            activation = _bn_relu(input)
            x = Conv3D(filters, kernel_size, strides=strides, padding="same",
                       use_bias=False, kernel_initializer="he_normal",
                       kernel_regularizer=l2(1e-4), name=name + "conv{}".format(flag))(activation)

        return x
    return f


def grouped_convolution(input, nb_channels, cardinality, name, strides=(1, 1, 1)):
    # when `cardinality` == 1 this is just a standard convolution
    if cardinality == 1:
        return Conv3D(nb_channels, kernel_size=(3, 3, 3), strides=strides,
                      padding='same', name=name + "groupConv")(input)

    assert not nb_channels % cardinality
    _d = nb_channels // cardinality

    # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
    # and convolutions are separately performed within each group
    groups = []
    for j in range(cardinality):
        group = Lambda(lambda z: z[:, :, :, :, j * _d:j * _d + _d])(input)
        groups.append(Conv3D(_d, kernel_size=(3, 3, 3), strides=strides,
                             padding='same', name=name + "groupConv_{}".format(j+1))(group))

    # the grouped convolutional layer concatenates them as the outputs of the layer
    input = concatenate(groups)

    return input


def _bn_relu_groupConv3D(filters, name, flag, groups):
    def f(input):
#       c = filters // groups
        activation = _bn_relu(input)
        x = grouped_convolution(activation, filters, groups, name)
        return x

#        x = Conv3D(filters, kernel_size=(3, 3, 3), strides=(1, 1, 1),
#                   padding="same", kernel_regularizer=l2(1e-4),
#                   kernel_initializer="he_normal", use_bias=False,
#                   name=name + "groupConv{}_{}".format(flag, flag))(activation)

        # 1x1x1 Conv3D
#        x = Conv3D(filters*c, kernel_size=(1, 1, 1), use_bias=False,
#                   kernel_initializer="he_normal", kernel_regularizer=l2(1e-4),
#                   name=name+"groupConv{}_{}".format(flag, flag+1))(x)

#        x_shape = K.int_shape(x)[1:-1]
#        x = Reshape(x_shape + (groups, c, c))(x)
#        x = Lambda(lambda x: sum([x[:, :, :, :, :, i] for i in range(c)]))(x)
#        x = Reshape(x_shape + (filters, ))(x)
#        return x

    return f


def unit_bottleneck(base_name, num_filters, input_tensor, layer_of_block, is_first_layer_of_first_block):
    '''
    Implement ResNeXt Fig.3(c)
    '''
    if layer_of_block == 1 and not is_first_layer_of_first_block:
        strides = (2, 2, 2)
    else:
        strides = (1, 1, 1)
    x = _bn_relu_Conv3D(num_filters[0], (1, 1, 1), base_name, 1, is_first_layer_of_first_block, strides)(input_tensor)
    x = _bn_relu_groupConv3D(num_filters[1], base_name, 2, 32)(x)
    #x = _bn_relu_Conv3D(num_filters[1], (3, 3, 3), base_name, 2, False, strides=(1, 1, 1))(x)
    x = _bn_relu_Conv3D(num_filters[2], (1, 1, 1), base_name, 3, False, strides=(1, 1, 1))(x)

    return x



def stage(stage_id, num_unit, num_filters, input):
    base_name = "stage%d_unit%d_"

    for i in range(1, num_unit + 1):
        if stage_id == 1 and i == 1:
            is_first_layer_of_first_block = True
            strides = (1, 1, 1)
        else:
            is_first_layer_of_first_block = False
            strides = (2, 2, 2)

        x = input
        x = unit_bottleneck(base_name % (stage_id, i), num_filters, x, i, is_first_layer_of_first_block)
        x = squeeze_excite_block(x)

        if i == 1:
            shortcut = Conv3D(num_filters[2], kernel_size=(1, 1, 1), strides=strides,
                              use_bias=False, name=base_name % (stage_id, i) + "sc")(input)
        else:
            shortcut = input

        input = add([x, shortcut])

    return input


def SEResNext50(input_shape, classes):
    input_tensor = Input(input_shape, name="input")
    x = Conv3D(64, (7, 7, 7), strides=(2, 2, 2), padding="same",
               use_bias=False, kernel_initializer="he_normal",
               kernel_regularizer=l2(1e-4), name="conv1")(input_tensor)
    x = _bn_relu(x)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding="same")(x)

    x = stage(1, 3, [128, 128, 256], x)
    x = stage(2, 4, [256, 256, 512], x)
    x = stage(3, 6, [512, 512, 1024], x)
    x = stage(4, 3, [1024, 1024, 2048], x)

    #x = stage(1, 3, [64, 64, 256], x)
    #x = stage(2, 4, [128, 128, 512], x)
    #x = stage(3, 6, [256, 256, 1024], x)
    #x = stage(4, 3, [512, 512, 2048], x)
    #x = GlobalAveragePooling3D()(x)

    x = _bn_relu(x)
    x_shape = K.int_shape(x)[1:-1]
    x = AveragePooling3D(pool_size=x_shape, strides=(1, 1, 1))(x)
    x = Flatten()(x)

    x = Dense(classes, activation="softmax", kernel_initializer="he_normal",
              kernel_regularizer=l2(1e-4))(x)
    model = Model(input_tensor, x)
    return model

