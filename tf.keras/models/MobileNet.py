from tensorflow.keras.layers import ZeroPadding3D, BatchNormalization, Conv3D, Activation
from tensorflow.keras.regularizers import l2
from fazekas_AI.DWConv import DepthwiseConv3D


def V1(input_tensor, filters=32, groups=32, kernel_size=1, strides=1, channel_axis=4, weighted_decay=5e-4):

    def _initial_conv_block(inputs, filters, kernel_size, strides):
        x = ZeroPadding3D(padding=(1, 1, 1))(inputs)
        x = Conv3D(filters=filters, kernel_size=kernel_size*3, strides=strides*2, padding='valid', use_bias=False,
                   kernel_initializer='he_normal', kernel_regularizer=l2(weighted_decay))(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        return x

    # depthwise convolution
    def _DWconv(inputs, groups, kernel_size, strides, padding):
        x = DepthwiseConv3D(groups=groups, kernel_size=kernel_size, strides=strides, padding=padding,
                            use_bias=False, depthwise_initializer='he_normal',
                            depthwise_regularizer=l2(weighted_decay))(inputs)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        return x

    # pointwise convolution
    def _PWconv(inputs, filters, kernel_size, strides, padding):
        x = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False,
                   kernel_initializer='he_normal', kernel_regularizer=l2(weighted_decay))(inputs)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        return x

    x = _initial_conv_block(input_tensor, filters=filters, kernel_size=kernel_size, strides=strides)

    x = _DWconv(x, groups=groups, kernel_size=kernel_size * 3, strides=strides, padding='same')
    x = _PWconv(x, filters=filters * 2, kernel_size=kernel_size, strides=strides, padding='same')
    x = ZeroPadding3D(padding=1)(x)

    x = _DWconv(x, groups=groups * 2, kernel_size=kernel_size * 3, strides=(2, 1, 2), padding='valid')
    x = _PWconv(x, filters=filters * 4, kernel_size=kernel_size, strides=strides, padding='same')

    x = _DWconv(x, groups=groups * 4, kernel_size=kernel_size * 3, strides=strides, padding='same')
    x = _PWconv(x, filters=filters * 4, kernel_size=kernel_size, strides=strides, padding='same')
    x = ZeroPadding3D(padding=1)(x)

    x = _DWconv(x, groups=groups * 4, kernel_size=kernel_size * 3, strides=(2, 1, 2), padding='valid')
    x = _PWconv(x, filters=filters * 8, kernel_size=kernel_size, strides=strides, padding='same')

    x = _DWconv(x, groups=groups * 8, kernel_size=kernel_size * 3, strides=strides, padding='same')
    x = _PWconv(x, filters=filters * 8, kernel_size=kernel_size, strides=strides, padding='same')
    x = ZeroPadding3D(padding=1)(x)

    x = _DWconv(x, groups=groups * 8, kernel_size=kernel_size * 3, strides=strides * 2, padding='valid')
    x = _PWconv(x, filters=filters * 16, kernel_size=kernel_size, strides=strides, padding='same')

    for i in range(5):
        x = _DWconv(x, groups=groups * 16, kernel_size=kernel_size * 3, strides=strides, padding='same')
        x = _PWconv(x, filters=filters * 16, kernel_size=kernel_size, strides=strides, padding='same')
    x = ZeroPadding3D(padding=1)(x)

    x = _DWconv(x, groups=groups * 16, kernel_size=kernel_size * 3, strides=strides * 2, padding='valid')
    x = _PWconv(x, filters=filters * 32, kernel_size=kernel_size, strides=strides, padding='same')

    x = _DWconv(x, groups=groups * 32, kernel_size=kernel_size * 3, strides=strides * 2, padding='same')
    x = _PWconv(x, filters=filters * 32, kernel_size=kernel_size, strides=strides, padding='same')

    return x