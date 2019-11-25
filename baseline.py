from tensorflow.keras.layers import ZeroPadding3D, BatchNormalization, Conv3D, Activation, add, Dense, Flatten, GlobalAveragePooling3D, Dropout
from tensorflow.keras.regularizers import l2

def BaseNet(input_tensor=None,
            filters=32,
            kernel_size=1,
            strides=1,
            channel_axis=4,
            weighted_decay=5e-4):

    def _conv_block(x, filters, strides):
        # x = ZeroPadding3D(padding=1)(x)
        x = Conv3D(filters=filters, kernel_size=kernel_size * 3, strides=strides, padding='same',
                   use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weighted_decay))(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        return x

    def _residual_block(x, filters):
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        x = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                   use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weighted_decay))(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        x = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                   use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weighted_decay))(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        x = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                   use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weighted_decay))(x)
        return x

    def _baselineNet(x, filters=[32, 64, 128, 256, 256, 128, 64, 32], name=None):
        x = _conv_block(x, filters[0], 1)

        init_x = _conv_block(x, filters[1], (2, 1, 2))
        x = _residual_block(init_x, filters[1])
        x = add([init_x, x])

        init_x = _conv_block(x, filters[2], (2, 2, 2))
        x = _residual_block(init_x, filters[2])
        x = add([init_x, x])

        init_x = _conv_block(x, filters[3], (2, 2, 2))
        x = _residual_block(init_x, filters[3])
        x = add([init_x, x])

        init_x = _conv_block(x, filters[4], (2, 2, 2))
        x = _residual_block(init_x, filters[4])
        x = add([init_x, x])

        init_x = _conv_block(x, filters[5], (2, 1, 2))
        x = _residual_block(init_x, filters[5])
        x = add([init_x, x])

        x = _conv_block(x, filters[6], 2)
        x = _conv_block(x, filters[7], 1)

        x = Flatten()(x)
        # x = GlobalAveragePooling3D()(x)

        x = Dense(200, 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weighted_decay), use_bias=False)(x)
        x = Dropout(0.2)(x)
        x = Dense(4, 'softmax', name=name)(x)

        return x

    out = _baselineNet(input_tensor, name='CNN')

    return out
