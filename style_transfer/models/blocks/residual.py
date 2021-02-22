import tensorflow as tf
import tensorflow_addons as tfa

from .convolution_layers import ConvolutionBlock


class ResidualBlock(tf.keras.Model):

    def __init__(self, channels, strides=1):
        super(ResidualBlock, self).__init__()
        self.conv_1 = ConvolutionBlock(
            channels, kernel_size=3, strides=strides)
        self.norm_1 = tfa.layers.InstanceNormalization()
        self.conv_2 = ConvolutionBlock(
            channels, kernel_size=3, strides=strides)
        self.norm_2 = tfa.layers.InstanceNormalization()

    def call(self, inputs, *args, **kwargs):
        residual = inputs
        x = self.conv_1(inputs)
        x = self.norm_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.norm_2(x)
        x = x + residual
        return x
