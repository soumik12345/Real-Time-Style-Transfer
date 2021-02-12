import tensorflow as tf


class StyleContentModel(tf.keras.models.Model):

    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        vgg = tf.keras.applications.VGG16(include_top=False, weights="imagenet")
        vgg.trainable = False
        style_outputs = [vgg.get_layer(name).output for name in style_layers]
        content_outputs = [
            vgg.get_layer(name).output for name in content_layers
        ]
        self.vgg = tf.keras.Model(
            [vgg.input], [style_outputs, content_outputs]
        )
        self.vgg.trainable = False

    def call(self, inputs, *args, **kwargs):
        preprocessed_input = tf.keras.applications.vgg16.preprocess_input(inputs)
        style_outputs, content_outputs = self.vgg(preprocessed_input)
        return style_outputs, content_outputs
