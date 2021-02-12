import tensorflow as tf


class ReflectionPadding2D(tf.keras.layers.Layer):

    def __init__(self, padding=1, **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.padding = padding

    def compute_output_shape(self, s):
        return s[0], s[1] + 2 * self.padding, s[2] + 2 * self.padding, s[3]

    def call(self, inputs, *args, **kwargs):
        return tf.pad(
            inputs, paddings=[
                [0, 0],
                [self.padding, self.padding],
                [self.padding, self.padding],
                [0, 0],
            ],
            mode="REFLECT",
        )
