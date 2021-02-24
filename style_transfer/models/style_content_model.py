import tensorflow as tf
from ..utils import gram_matrix


def _get_vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model


class StyleContentModel(tf.keras.models.Model):

    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = _get_vgg_layers(style_layers + content_layers)
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs, *args, **kwargs):
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (
            outputs[:self.num_style_layers],
            outputs[self.num_style_layers:]
        )
        # Compute the gram_matrix
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        # Features that extracted by VGG
        style_dict = {
            style_name: value for style_name, value in zip(self.style_layers, style_outputs)
        }
        content_dict = {
            content_name: value for content_name, value in zip(self.content_layers, content_outputs)
        }
        return {
            'content': content_dict,
            'style': style_dict
        }
