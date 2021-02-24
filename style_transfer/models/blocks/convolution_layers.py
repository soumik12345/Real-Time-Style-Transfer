import tensorflow as tf

from .instance_normalization import InstanceNormalization


class ConvolutionBlock(tf.keras.layers.Layer):

    def __init__(self, filters, kernel, stride):
        super(ConvolutionBlock, self).__init__()
        pad = kernel // 2
        self.paddings = tf.constant([
            [0, 0], [pad, pad], [pad, pad], [0, 0]
        ])
        self.conv2d = tf.keras.layers.Conv2D(
            filters, kernel, stride,
            use_bias=False, padding='valid'
        )
        self.instance_norm = InstanceNormalization()

    def call(self, inputs, relu: bool = True, *args, **kwargs):
        x = tf.pad(inputs, self.paddings, mode='REFLECT')
        x = self.conv2d(x)
        x = self.instance_norm(x)
        if relu:
            x = tf.nn.relu(x)
        return x


class UpsampleBlock(tf.keras.layers.Layer):

    def __init__(self, filters, kernel, stride):
        super(UpsampleBlock, self).__init__()
        self.tran_conv = tf.keras.layers.Conv2DTranspose(
            filters, kernel, stride, padding='same'
        )
        self.instance_norm = InstanceNormalization()

    def call(self, inputs, *args, **kwargs):
        x = self.tran_conv(inputs)
        x = self.instance_norm(x)
        return tf.nn.relu(x)


class ResizeConvolutionBlock(tf.keras.layers.Layer):

    def __init__(self, filters, kernel, stride):
        super(ResizeConvolutionBlock, self).__init__()
        self.conv = ConvolutionBlock(filters, kernel, stride)
        self.instance_norm = InstanceNormalization()
        self.stride = stride

    def call(self, inputs, *args, **kwargs):
        new_height = inputs.shape[1] * self.stride * 2
        new_width = inputs.shape[2] * self.stride * 2
        x = tf.image.resize(
            inputs, [new_height, new_width],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        x = self.conv(x)
        x = self.instance_norm(x)
        return tf.nn.relu(x)


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, filters, kernel, stride):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvolutionBlock(filters, kernel, stride)
        self.conv2 = ConvolutionBlock(filters, kernel, stride)

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs)
        return inputs + self.conv2(x, relu=False)
