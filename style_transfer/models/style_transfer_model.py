from typing import List
import tensorflow as tf

from ..utils import read_image
from .transformer_model import TransformerModel
from .style_content_model import StyleContentModel
from style_transfer.models.loss import style_loss, content_loss, total_variation_loss


class StyleTransferModel(tf.keras.Model):

    def __init__(
            self, content_layers: List[str], style_layers: List[str],
            style_image_file: str, sample_content_image_file: str,
            image_size: int, batch_size: int,
            style_weight: float, content_weight: float, total_variation_weight: float):
        super(StyleTransferModel, self).__init__()
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.image_size = image_size
        self.batch_size = batch_size
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.total_variation_weight = total_variation_weight
        self.optimizer = None
        self.x_batch = tf.zeros(
            (batch_size, image_size, image_size, 3), dtype=tf.float32
        )
        self.style_image = read_image(
            image_file=style_image_file, image_size=image_size
        )
        self.sample_content_image = read_image(
            image_file=sample_content_image_file, image_size=image_size
        )
        self.transformer_model = TransformerModel()
        self.feature_extractor = StyleContentModel(
            style_layers=self.style_layers,
            content_layers=self.content_layers
        )
        self.style_target = None

    def compile(self, optimizer, **kwargs):
        self.optimizer = optimizer
        self.style_target = self.feature_extractor(self.style_image)['style']

    def train_step(self, data):

        with tf.GradientTape() as tape:

            content_target = self.feature_extractor(self.x_batch)['content']
            image = self.transformer_model(self.x_batch)
            outputs = self.feature_extractor(image)

            _style_loss = self.style_weight * style_loss(
                style_outputs=outputs['style'],
                style_target=self.style_target
            )
            _content_weight = self.content_weight * content_loss(
                content_outputs=outputs['content'],
                content_target=content_target
            )
            _tv_loss = self.total_variation_weight * total_variation_loss(image=image)
            total_loss = _style_loss + _content_weight + _tv_loss

        gradients = tape.gradient(
            total_loss, self.transformer_model.trainable_variables
        )
        self.optimizer.apply_gradients(
            zip(gradients, self.transformer_model.trainable_variables)
        )
        return {
            'style_loss': _style_loss,
            'content_loss': _content_weight,
            'total_variation_loss': _tv_loss,
            'total_loss': total_loss
        }

    def call(self, inputs, training=False, *args, **kwargs):
        return self.transformer_model(inputs, training=training)

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self.transformer_model.save_weights(
            filepath=filepath, overwrite=overwrite,
            save_format=save_format, options=options
        )
