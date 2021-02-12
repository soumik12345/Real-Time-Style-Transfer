import tensorflow as tf
import tensorflow_addons as tfa

from .blocks import ConvolutionBlock, UpsampleBlock, ResidualBlock


class TransformerNet(tf.keras.Model):

    def __init__(self):
        super(TransformerNet, self).__init__()
        self.conv1 = ConvolutionBlock(32, kernel_size=9, strides=1)
        self.norm1 = tfa.layers.InstanceNormalization()
        self.conv2 = ConvolutionBlock(64, kernel_size=3, strides=2)
        self.norm2 = tfa.layers.InstanceNormalization()
        self.conv3 = ConvolutionBlock(128, kernel_size=3, strides=2)
        self.norm3 = tfa.layers.InstanceNormalization()

        self.res_blocks = [ResidualBlock(channels=128) for _ in range(5)]

        self.upsample1 = UpsampleBlock(
            64, kernel_size=3, strides=1, upsample=2
        )
        self.norm4 = tfa.layers.InstanceNormalization()
        self.upsample2 = UpsampleBlock(
            32, kernel_size=3, strides=1, upsample=2
        )
        self.norm5 = tfa.layers.InstanceNormalization()
        self.upsample3 = ConvolutionBlock(3, kernel_size=9, strides=1)

    def call(self, inputs, *args, **kwargs):
        x = self.relu(self.norm1(self.conv1(inputs)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.relu(self.norm3(self.conv3(x)))
        for res_block in self.res_blocks:
            x = res_block(x)
        x = tf.nn.relu(self.norm4(self.upsample1(x)))
        x = tf.nn.relu(self.norm5(self.upsample2(x)))
        x = self.upsample3(x)
        return x
