import tensorflow as tf


class InstanceNormalization(tf.keras.layers.Layer):

    def __init__(self, epsilon=1e-3):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
        self.beta, self.gamma = None, None

    def build(self, input_shape):
        self.beta = tf.Variable(tf.zeros([input_shape[3]]))
        self.gamma = tf.Variable(tf.ones([input_shape[3]]))

    def call(self, inputs, *args, **kwargs):
        mean, var = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        x = tf.divide(
            tf.subtract(inputs, mean),
            tf.sqrt(tf.add(var, self.epsilon))
        )
        return self.gamma * x + self.beta
