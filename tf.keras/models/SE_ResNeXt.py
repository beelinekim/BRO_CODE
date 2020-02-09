from tensorflow.keras.layers import Add, Activation, GlobalAveragePooling3D, Dense, Conv3D, BatchNormalization
from tensorflow.keras.layers import MaxPooling3D, Reshape, Multiply, concatenate, add
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K


def ResNext50(inputs, weighted_decay=5e-4, cardinality=8, depth=29, width=4):
    """
    ResNext50 is [3, 4, 6, 3]
    ResNext101 is [3, 4, 23, 3]
    ResNext152 is [3, 8, 23, 3]
    """

    # inputs = Input()
    N = [(depth - 2) // 9 for _ in range(3)]

    filters = cardinality * width
    filters_list = []

    for i in range(len(N)):
        filters_list.append(filters)
        filters *= 2

    x = _initial_conv_block(inputs, weighted_decay=weighted_decay)

    for i in range(N[0]):
        x = _bottleneck_block(x, filters_list[0], cardinality, strides=1, weighted_decay=weighted_decay)

    N = N[1:]
    filters_list = filters_list[1:]

    for block_idx, n_i in enumerate(N):
        for j in range(n_i):
            if i == 0:
                x = _bottleneck_block(x, filters_list[block_idx], cardinality, strides=2, weighted_decay=weighted_decay)
            else:
                x = _bottleneck_block(x, filters_list[block_idx], cardinality, strides=1, weighted_decay=weighted_decay)

    x = GlobalAveragePooling3D()(x)

    return x


def _initial_conv_block(inputs, filters=64, kernel_size=7, strides=2, weighted_decay=5e-4):
    channel_axis = 3

    x = Conv3D(filters, kernel_size, strides, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weighted_decay))(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(3, strides)(x)

    return x


def _grouped_conv_block(inputs, grouped_channels, kernel_size=1, strides=1, weighted_decay=5e-4, cardinality=32):
    group_list = []
    channel_axis = 3

    for i in range(cardinality):
        x = Conv3D(grouped_channels, kernel_size, strides, padding='same', use_bias=False,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(weighted_decay))(inputs)
        x = Conv3D(grouped_channels, kernel_size * 3, strides, padding='same', use_bias=False,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(weighted_decay))(x)
        x = Conv3D(grouped_channels, kernel_size, strides, padding='same', use_bias=False,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(weighted_decay))(x)

        group_list.append(x)

    group_merge = concatenate(group_list, axis=channel_axis)
    x = BatchNormalization(axis=channel_axis)(group_merge)
    x = Activation('relu')(x)

    return x


def _bottleneck_block(x, filters=64, kernel_size=1, strides=1, weighted_decay=5e-4, cardinality=8):
    init = x
    grouped_channels = int(filters / cardinality)
    channel_axis = 3

    if K.image_data_format() == 'channels_first':
        if init._keras_shape[1] != 2 * filters:
            init = Conv3D(filters * 2, kernel_size, strides, padding='same', use_bias=False,
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(weighted_decay))(init)
            init = BatchNormalization(axis=channel_axis)(init)
    else:
        if init._keras_shape[-1] != 2 * filters:
            init = Conv3D(filters * 2, kernel_size, strides, padding='same', use_bias=False,
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(weighted_decay))(init)
            init = BatchNormalization(axis=channel_axis)(init)

    x = Conv3D(filters, kernel_size, strides, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weighted_decay))(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = _grouped_conv_block(x, grouped_channels, cardinality, strides, weighted_decay)

    x = Conv3D(filters * 2, 1, strides, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weighted_decay))(x)
    x = BatchNormalization(axis=channel_axis)(x)

    # x = squeeze_excite_block(x, filters * 2)

    x = add([init, x])
    x = Activation('relu')(x)

    return x


def squeeze_excite_block(x, filters=256, ratio=16):
    init = x
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling3D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', use_bias=False, kernel_initializer='he_normal')(se)
    se = Dense(filters, activation='sigmoid', use_bias=False, kernel_initializer='he_normal')(se)

    x = Multiply([init, se])
    return x