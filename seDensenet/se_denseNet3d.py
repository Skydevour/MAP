from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Conv3D,
    AveragePooling3D,
    MaxPooling3D,
    GlobalAveragePooling3D,
    Dropout,
    Concatenate,
    multiply
)
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
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
    #se_shape = (1, 1, 1, filters)
    #se = GlobalAveragePooling3D()(init)
    #se = Reshape(se_shape)(se)
    #se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    #se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    se = AveragePooling3D(K.int_shape(init)[1:-1])(init)
    #se = Flatten()(se)
    se = Conv3D(filters//ratio, kernel_size=(1, 1, 1), kernel_initializer='he_normal', use_bias=False)(se)
    se = Activation("relu")(se)
    se = Conv3D(filters, kernel_size=(1, 1, 1), kernel_initializer='he_normal', use_bias=False)(se)
    se = Activation("sigmoid")(se)
    x = multiply([init, se])
    return x


def _bn_relu(x, concat_axis):
    """ Apply BatchNorm, Relu"""
    x = BatchNormalization(axis=concat_axis)(x)
    x = Activation("relu")(x)
    return x


def _bn_relu_conv3d(x, concat_axis, nb_filter,
                    dropout_rate=None, weight_decay=1e-4):
    """Apply BatchNorm, Relu 3x3x3Conv3D, optional dropout
        :param x: Input keras network
        :param concat_axis: int -- index of contatenate axis
        :param nb_filter: int -- number of filters
        :param dropout_rate: int -- dropout rate
        :param weight_decay: int -- weight decay factor

        :returns: keras network with b_norm, relu and Conv3D added
        :rtype: keras network
    """

    x = _bn_relu(x, concat_axis)
    x = Conv3D(filters=4*nb_filter, kernel_size=(1, 1, 1),
               strides=(1, 1, 1), kernel_initializer="glorot_uniform",
               padding="same", use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = Conv3D(filters=nb_filter, kernel_size=(3, 3, 3),
               strides=(1, 1, 1), kernel_initializer="glorot_uniform",
               padding="same", use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition(x, concat_axis, reduction=0.5,
               dropout_rate=None, weight_decay=1e-4):
    """Apply BatchNorm, Relu 1x1x1Conv3D, optional dropout and Maxpooling3D
        :param x: keras model
        :param concat_axis: int -- index of contatenate axis
        :param nb_filter: int -- number of filters
        :param dropout_rate: int -- dropout rate
        :param weight_decay: int -- weight decay factor

        :returns: model
        :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool
    """

    x = _bn_relu(x, concat_axis)
    nb_filter = K.int_shape(x)[-1]
    x = Conv3D(filters=int(nb_filter*reduction), kernel_size=(1, 1, 1),
               strides=(1, 1, 1), kernel_initializer="glorot_uniform",
               padding="same", use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)

    return x


def denseblock(x, concat_axis, nb_layers, nb_filter, growth_rate,
               dropout_rate=None, weight_decay=1e-4):
    """Build a denseblock where the output of each
           _bn_relu_conv3d is fed to subsequent ones
        :param x: keras model
        :param concat_axis: int -- index of contatenate axis
        :param nb_layers: int -- the number of layers of _bn_relu_conv3d
                                    to append to the model.
        :param nb_filter: int -- number of filters
        :param dropout_rate: int -- dropout rate
        :param weight_decay: int -- weight decay factor

        :returns: keras model with nb_layers of _bn_relu_conv3d appended
        :rtype: keras model
    """

    for i in range(nb_layers):
        merge_tensor = _bn_relu_conv3d(x, concat_axis, growth_rate,
                                       dropout_rate, weight_decay)
        x = Concatenate(axis=concat_axis)([merge_tensor, x])
        nb_filter += growth_rate

    x = squeeze_excite_block(x)
    return x, nb_filter


def DenseNet(img_dim, nb_classes, depth=40, nb_dense_block=3, growth_rate=12,
             reduction=0.5, nb_filter=64, dropout_rate=None, weight_decay=1e-4):
    """ Build the DenseNet model
        :param nb_classes: int -- number of classes
        :param img_dim: tuple -- (channels, rows, columns)
        :param depth: int -- how many layers
        :param nb_dense_block: int -- number of dense blocks to add to end
        :param growth_rate: int -- number of filters to add
        :param nb_filter: int -- number of filters
        :param dropout_rate: float -- dropout rate
        :param weight_decay: float -- weight decay

        :returns: keras model with nb_layers of _bn_relu_conv appended
        :rtype: keras model
    """

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    if len(img_dim) != 4:
        raise ValueError("Input shape should be a tuple "
                         "(conv_dim1, conv_dim2, conv_dim3, channels) "
                         "for tensorflow as backend or "
                         "(channels, conv_dim1, conv_dim2, conv_dim3) "
                         "for theano as backend")

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    model_input = Input(shape=img_dim)

    # layers in each dense block
    nb_layers = (depth - 4) // 3

    # Initial convolution
    x = Conv3D(filters=nb_filter, kernel_size=(3, 3, 3),
               strides=(1, 1, 1), kernel_initializer="glorot_uniform",
               padding="same", use_bias=False,
               kernel_regularizer=l2(weight_decay), name="inital_conv3D")(model_input)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = denseblock(x, concat_axis, nb_layers,
                                  nb_filter, growth_rate,
                                  dropout_rate, weight_decay)

        # add transition
        x = transition(x, concat_axis, reduction, dropout_rate, weight_decay)

    # The last denseblock does not have a transition
    x, nb_filter = denseblock(x, concat_axis, nb_layers, nb_filter, growth_rate,
                   dropout_rate, weight_decay)

    x = _bn_relu(x, concat_axis)
    x = GlobalAveragePooling3D()(x)
    x = Dense(nb_classes, activation="softmax",
              kernel_initializer="glorot_uniform",
              use_bias=False,
              kernel_regularizer=l2(weight_decay))(x)
    densenet = Model(inputs=model_input, outputs=x, name="DenseNet")

    return densenet



def denseNet22(img_dim, nb_classes, depth=22, nb_dense_block=3, growth_rate=8,
               reduction=0.5, nb_filter=10, dropout_rate=None, weight_decay=1e-4):

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"
    model_input = Input(shape=img_dim)

    nb_layers = (depth - 4) // 3
    x = Conv3D(filters=nb_filter, kernel_size=(3, 3, 3),
               strides=(1, 1, 1), kernel_initializer="glorot_uniform",
               padding="same", use_bias=False,
               kernel_regularizer=l2(weight_decay))(model_input)

    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = denseblock(x, concat_axis, nb_layers,
                                  nb_filter, growth_rate,
                                  dropout_rate, weight_decay)
        # add transition
        x = transition(x, concat_axis, reduction, dropout_rate, weight_decay)

    # The last denseblock does not have a transition
    x, nb_filter = denseblock(x, concat_axis, nb_layers, nb_filter, growth_rate,
                              dropout_rate, weight_decay)

    x = _bn_relu(x, concat_axis)
    x = AveragePooling3D(K.int_shape(x)[1:-1])(x)
    x = Flatten()(x)
    x = Dense(nb_classes, activation="softmax",
              kernel_initializer="glorot_uniform",
              use_bias=False,
              kernel_regularizer=l2(weight_decay))(x)

    densenet = Model(inputs=model_input, outputs=x, name="denseNet22")
    return densenet