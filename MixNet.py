# https://github.com/titu1994

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras import layers
from tensorflow.keras import models
from keras.utils import conv_utils
from keras.utils.generic_utils import get_custom_objects
from fazekas_AI.DWConv import DepthwiseConv3D

class MixNetConvInitializer(initializers.Initializer):
    def __init__(self):
        super(MixNetConvInitializer, self).__init__()

    def __call__(self, shape, dtype=None):
        dtype = dtype or K.floatx()

        kernel_height, kernel_width, _, out_filters = shape
        fan_out = int(kernel_height * kernel_width * out_filters)
        return tf.random_normal(
            shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)

class DropConnect(layers.Layer):

    def __init__(self, drop_connect_rate=0.):
        super(DropConnect, self).__init__()
        self.drop_connect_rate = float(drop_connect_rate)

    def call(self, inputs, training=None):

        def drop_connect():
            keep_prob = 1.0 - self.drop_connect_rate

            # Compute drop_connect tensor
            batch_size = tf.shape(inputs)[0]
            random_tensor = keep_prob
            random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            output = (inputs / keep_prob) * binary_tensor
            return output

        return K.in_train_phase(drop_connect, inputs, training=training)

def _split_channels(total_filters, groups):
    split = [total_filters // groups for _ in range(groups)]
    split[0] += total_filters - sum(split)

class GroupConv3D(models.Model):
    def __init__(self, filters, kernels, groups):
        super(GroupConv3D, self).__init__()

        self.filters = filters
        self.kernels = kernels
        self.groups = groups
        self.strides = 1
        self.padding = 'same'
        self.use_bias = False

        self._layers = [DepthwiseConv3D(kernels[i],
                                        strides=self.strides,
                                        padding=self.padding,
                                        use_bias=self.use_bias,
                                        kernel_initializer=MixNetConvInitializer())
                        for i in range(groups)]

        self.data_format = 'channels_last'
        self._channel_axis = 4

    def call(self, inputs, **kwargs):
        if len(self._layers) == 1:
            return self._layers[0](inputs)

        filters = K.int_shape(inputs)[self._channel_axis]
        splits = _split_channels(filters, self.groups)
        x_splits = tf.split(inputs, splits, self._channel_axis)
        x_outputs = [c(x) for x, c in zip(x_splits, self._layers)]
        x = layers.concatenate(x_outputs, axis=self._channel_axis)
        return x

class GroupedConv3D(object):
    def __init__(self, filters, kernel_size):
        self._groups = len(kernel_size)
        self._channel_axis = 4
        self.filters = filters
        self.kernels = kernel_size

    def __call__(self, inputs):
        grouped_op = GroupConv3D(filters=self.filters, kernels=self.kernels, groups=self._groups)
        x = grouped_op(inputs)
        return x

def SEBlock(input_filters, se_ratio, expand_ratio, activation_fn, data_format=None):
    if data_format is None:
        data_format = K.image_data_format()

    num_reduced_filters = max(
        1, int(input_filters * se_ratio))
    filters = input_filters * expand_ratio

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    def block(inputs):
        x = inputs
        x = layers.Lambda(lambda a: K.mean(a, axis=spatial_dims, keepdims=True))(x)
        x = GroupedConv2D(
            num_reduced_filters,
            kernel_size=[1],
            strides=[1, 1],
            kernel_initializer=MixNetConvInitializer(),
            padding='same',
            use_bias=True)(x)

        x = activation_fn()(x)

        # Excite
        x = GroupedConv2D(
            filters,
            kernel_size=[1],
            strides=[1, 1],
            kernel_initializer=MixNetConvInitializer(),
            padding='same',
            use_bias=True)(x)
        x = layers.Activation('sigmoid')(x)
        out = layers.Multiply()([x, inputs])
        return out

    return block


# Obtained from
class MDConv(object):

    def __init__(self, kernel_size, strides, dilated=False, **kwargs):
        self._channel_axis = -1
        self._dilated = dilated
        self.kernels = kernel_size

        self._conv_kwargs = {
            'strides': strides,
            'dilation_rate': kwargs.get('dilation_rate', (1, 1)),
            'kernel_initializer': kwargs.get('kernel_initializer', MixNetConvInitializer()),
            'padding': 'same',
            'use_bias': kwargs.get('use_bias', False),
        }

    def __call__(self, inputs):
        filters = K.int_shape(inputs)[self._channel_axis]
        grouped_op = GroupConvolution(filters, self.kernels, groups=len(self.kernels),
                                      type='depthwise_conv', conv_kwargs=self._conv_kwargs)
        x = grouped_op(inputs)
        return x


# Obtained from https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/mixnet_model.py
def MixNetBlock(input_filters, output_filters,
                dw_kernel_size, expand_kernel_size,
                project_kernel_size, strides,
                expand_ratio, se_ratio,
                id_skip, drop_connect_rate,
                batch_norm_momentum=0.99,
                batch_norm_epsilon=1e-3,
                swish=False,
                dilated=None,
                data_format=None):

    if data_format is None:
        data_format = K.image_data_format()

    if data_format == 'channels_first':
        channel_axis = 1
        spatial_dims = [2, 3]
    else:
        channel_axis = -1
        spatial_dims = [1, 2]

    has_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
    filters = input_filters * expand_ratio
    relu_activation = Swish if swish else layers.ReLU

    def block(inputs):
        # Expand part
        if expand_ratio != 1:
            x = GroupedConv2D(
                filters,
                kernel_size=expand_kernel_size,
                strides=[1, 1],
                kernel_initializer=MixNetConvInitializer(),
                padding='same',
                use_bias=False)(inputs)

            x = layers.BatchNormalization(
                axis=channel_axis,
                momentum=batch_norm_momentum,
                epsilon=batch_norm_epsilon)(x)

            x = relu_activation()(x)
        else:
            x = inputs

        kernel_size = dw_kernel_size
        # Depthwise Convolutional Phase
        x = MDConv(
            kernel_size,
            strides=strides,
            dilated=dilated,
            depthwise_initializer=MixNetConvInitializer(),
            padding='same',
            use_bias=False)(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)(x)
        x = relu_activation()(x)

        if has_se:
            x = SEBlock(input_filters, se_ratio, expand_ratio,
                        relu_activation,
                        data_format)(x)

        # output phase
        x = GroupedConv2D(
            output_filters,
            kernel_size=project_kernel_size,
            strides=[1, 1],
            kernel_initializer=MixNetConvInitializer(),
            padding='same',
            use_bias=False)(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon)(x)

        if id_skip:
            if all(s == 1 for s in strides) and (
                    input_filters == output_filters):

                # only apply drop_connect if skip presents.
                # if drop_connect_rate:
                #     x = DropConnect(drop_connect_rate)(x)

                x = layers.Add()([x, inputs])

        return x

    return block


def MixNet(input_shape,
           block_args_list: List[BlockArgs],
           depth_multiplier: float,
           include_top=True,
           weights=None,
           input_tensor=None,
           pooling=None,
           classes=1000,
           dropout_rate=0.,
           drop_connect_rate=0.,
           batch_norm_momentum=0.99,
           batch_norm_epsilon=1e-3,
           depth_divisor=8,
           stem_size=16,
           feature_size=1536,
           min_depth=None,
           data_format=None,
           default_size=None,
           **kwargs):

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` '
                         'as true, `classes` should be 1000')

    if data_format is None:
        data_format = K.image_data_format()

    if data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    if default_size is None:
        default_size = 224

    if block_args_list is None:
        block_args_list = get_mixnet_small()

    # count number of strides to compute min size
    stride_count = 1
    for block_args in block_args_list:
        if block_args.strides is not None and block_args.strides[0] > 1:
            stride_count += 1

    min_size = int(2 ** stride_count)

    # Determine proper input shape and default size.
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_size,
                                      min_size=min_size,
                                      data_format=data_format,
                                      require_flatten=include_top,
                                      weights=weights)

    # Stem part
    if input_tensor is None:
        inputs = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            inputs = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            inputs = input_tensor

    # Stem part
    x = inputs
    x = GroupedConv2D(
        filters=round_filters(stem_size, depth_multiplier,
                              depth_divisor, min_depth),
        kernel_size=[3],
        strides=[2, 2],
        kernel_initializer=MixNetConvInitializer(),
        padding='same',
        use_bias=False)(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon)(x)
    x = layers.ReLU()(x)

    num_blocks = sum([block_args.num_repeat for block_args in block_args_list])
    drop_connect_rate_per_block = drop_connect_rate / float(num_blocks)

    # Blocks part
    for block_idx, block_args in enumerate(block_args_list):
        assert block_args.num_repeat > 0

        # Update block input and output filters based on depth multiplier.
        block_args.input_filters = round_filters(block_args.input_filters, depth_multiplier, depth_divisor, min_depth)
        block_args.output_filters = round_filters(block_args.output_filters, depth_multiplier, depth_divisor, min_depth)
        block_args.num_repeat = round_repeats(block_args.num_repeat)

        # The first block needs to take care of stride and filter size increase.
        x = MixNetBlock(block_args.input_filters, block_args.output_filters,
                        block_args.dw_kernel_size, block_args.expand_kernel_size,
                        block_args.project_kernel_size, block_args.strides,
                        block_args.expand_ratio, block_args.se_ratio,
                        block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                        batch_norm_momentum, batch_norm_epsilon, block_args.swish,
                        block_args.dilated, data_format)(x)

        if block_args.num_repeat > 1:
            block_args.input_filters = block_args.output_filters
            block_args.strides = [1, 1]

        for _ in range(block_args.num_repeat - 1):
            x = MixNetBlock(block_args.input_filters, block_args.output_filters,
                            block_args.dw_kernel_size, block_args.expand_kernel_size,
                            block_args.project_kernel_size, block_args.strides,
                            block_args.expand_ratio, block_args.se_ratio,
                            block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                            batch_norm_momentum, batch_norm_epsilon, block_args.swish,
                            block_args.dilated, data_format)(x)

    # Head part
    x = GroupedConv2D(
        filters=feature_size,
        kernel_size=[1],
        strides=[1, 1],
        kernel_initializer=MixNetConvInitializer(),
        padding='same',
        use_bias=False)(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon)(x)
    x = layers.ReLU()(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(data_format=data_format)(x)

        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

        x = layers.Dense(classes, kernel_initializer=MixNetDenseInitializer())(x)
        x = layers.Activation('softmax')(x)

    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    outputs = x

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)

    model = Model(inputs, outputs)

    return model


def MixNetSmall(input_shape=None,
                include_top=True,
                weights=None,
                input_tensor=None,
                pooling=None,
                classes=1000,
                dropout_rate=0.2,
                drop_connect_rate=0.,
                data_format=None):

    return MixNet(input_shape,
                  get_mixnet_small(),
                  depth_multiplier=1.0,
                  include_top=include_top,
                  weights=weights,
                  input_tensor=input_tensor,
                  pooling=pooling,
                  classes=classes,
                  dropout_rate=dropout_rate,
                  drop_connect_rate=drop_connect_rate,
                  data_format=data_format,
                  default_size=224)


def MixNetMedium(input_shape=None,
                 include_top=True,
                 weights=None,
                 input_tensor=None,
                 pooling=None,
                 classes=1000,
                 dropout_rate=0.25,
                 drop_connect_rate=0.,
                 data_format=None):

    return MixNet(input_shape,
                  get_mixnet_medium(),
                  depth_multiplier=1.0,
                  include_top=include_top,
                  weights=weights,
                  input_tensor=input_tensor,
                  pooling=pooling,
                  classes=classes,
                  dropout_rate=dropout_rate,
                  drop_connect_rate=drop_connect_rate,
                  stem_size=24,
                  data_format=data_format,
                  default_size=224)


def MixNetLarge(input_shape=None,
                include_top=True,
                weights=None,
                input_tensor=None,
                pooling=None,
                classes=1000,
                dropout_rate=0.3,
                drop_connect_rate=0.,
                data_format=None):

    return MixNet(input_shape,
                  get_mixnet_large(),
                  depth_multiplier=1.3,
                  include_top=include_top,
                  weights=weights,
                  input_tensor=input_tensor,
                  pooling=pooling,
                  classes=classes,
                  dropout_rate=dropout_rate,
                  drop_connect_rate=drop_connect_rate,
                  stem_size=24,
                  data_format=data_format,
                  default_size=224)
