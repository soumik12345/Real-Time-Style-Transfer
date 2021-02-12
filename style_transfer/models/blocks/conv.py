import tensorflow as tf

from .reflection_padding import ReflectionPadding2D


class ConvolutionBlock(tf.keras.layers.Layer):

    def __init__(self, channels, kernel_size=3, strides=1):
        super(ConvolutionBlock, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = ReflectionPadding2D(reflection_padding)
        self.convolution = tf.keras.layers.Conv2D(
            channels, kernel_size, strides=strides
        )

    def call(self, inputs, *args, **kwargs):
        x = self.reflection_pad(inputs)
        x = self.convolution(x)
        return x


class UpsampleBlock(tf.keras.layers.Layer):

    def __init__(self, channels, kernel_size=3, strides=1, upsample=2):
        super(UpsampleBlock, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = ReflectionPadding2D(reflection_padding)
        self.convolution = tf.keras.layers.Conv2D(
            channels, kernel_size, strides=strides
        )
        self.upsample = tf.keras.layers.UpSampling2D(size=upsample)

    def call(self, inputs, *args, **kwargs):
        x = self.upsample(inputs)
        x = self.reflection_pad(x)
        x = self.convolution(x)
        return x
