import tensorflow as tf
from .blocks import (
    ConvolutionBlock, ResizeConvolutionBlock, ResidualBlock
)


class TransformerModel(tf.keras.models.Model):

    def __init__(self):
        super(TransformerModel, self).__init__()
        self.conv1 = ConvolutionBlock(32, 9, 1)
        self.conv2 = ConvolutionBlock(64, 3, 2)
        self.conv3 = ConvolutionBlock(128, 3, 2)
        self.residual1 = ResidualBlock(128, 3, 1)
        self.residual2 = ResidualBlock(128, 3, 1)
        self.residual3 = ResidualBlock(128, 3, 1)
        self.residual4 = ResidualBlock(128, 3, 1)
        self.residual5 = ResidualBlock(128, 3, 1)
        self.resize_conv1 = ResizeConvolutionBlock(64, 3, 2)
        self.resize_conv2 = ResizeConvolutionBlock(32, 3, 2)
        self.conv4 = ConvolutionBlock(3, 9, 1)

    def call(self, inputs, *args, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        x = self.residual4(x)
        x = self.residual5(x)
        x = self.resize_conv1(x)
        x = self.resize_conv2(x)
        x = self.conv4(x, relu=False)
        return tf.nn.tanh(x) * 150 + 255. / 2
