from tqdm import tqdm
from typing import List
import tensorflow as tf
from tqdm.notebook import tqdm as tqdm_notebook

from ..utils import read_image
from .transformer_model import TransformerModel
from .style_content_model import StyleContentModel
from style_transfer.models.loss import style_loss, content_loss, total_variation_loss


class StyleTransferModel:

    def __init__(
            self, content_layers: List[str], style_layers: List[str],
            style_image_file: str, sample_content_image_file: str,
            image_size: int, batch_size: int, experiment_name: str,
            style_weight: float, content_weight: float, total_variation_weight: float):
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.image_size = image_size
        self.batch_size = batch_size
        self.experiment_name = experiment_name
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.total_variation_weight = total_variation_weight
        self.optimizer = None
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
        self.style_target, self.loss_metrics = None, {}

    def _initialize_summary_writer(self, log_dir: str):
        self.summary_writer = tf.summary.create_file_writer(logdir=log_dir)
        with self.summary_writer.as_default():
            tf.summary.image('Train/style_image', self.style_image / 255.0, step=0)
            tf.summary.image('Train/content_image', self.sample_content_image / 255.0, step=0)

    def _initialize_metrics(self):
        self.loss_metrics = {
            'total_loss': tf.keras.metrics.Mean(name='total_loss'),
            'style_loss': tf.keras.metrics.Mean(name='style_loss'),
            'content_loss': tf.keras.metrics.Mean(name='content_loss'),
            'total_variation_loss': tf.keras.metrics.Mean(name='total_variation_loss')
        }

    def compile(self, optimizer, log_dir: str):
        self.optimizer = optimizer
        self.style_target = self.feature_extractor(self.style_image)['style']
        self._initialize_metrics()
        self._initialize_summary_writer(log_dir=log_dir)

    def _update_tensorboard(self, step: int):
        with self.summary_writer.as_default():
            tf.summary.scalar(
                'scalars/style_loss',
                self.loss_metrics['style_loss'].result(), step=step
            )
            tf.summary.scalar(
                'scalars/content_loss',
                self.loss_metrics['content_loss'].result(), step=step
            )
            tf.summary.scalar(
                'scalars/total_variation_loss',
                self.loss_metrics['total_variation_loss'].result(), step=step
            )
            tf.summary.scalar(
                'scalars/total_loss',
                self.loss_metrics['total_loss'].result(), step=step
            )
            sample_styled_image = self.transformer_model(self.sample_content_image)
            tf.summary.image(
                'Train/styled_image',
                sample_styled_image / 255.0, step=step
            )
        self.loss_metrics['style_loss'].reset_states()
        self.loss_metrics['content_loss'].reset_states()
        self.loss_metrics['total_variation_loss'].reset_states()
        self.loss_metrics['total_loss'].reset_states()

    @tf.function
    def train_step(self, data):

        with tf.GradientTape() as tape:

            content_target = self.feature_extractor(data)['content']
            image = self.transformer_model(data)
            outputs = self.feature_extractor(image)

            _style_loss = self.style_weight * style_loss(
                style_outputs=outputs['style'],
                style_target=self.style_target
            )
            _content_loss = self.content_weight * content_loss(
                content_outputs=outputs['content'],
                content_target=content_target
            )
            _tv_loss = self.total_variation_weight * total_variation_loss(image=image)
            total_loss = _style_loss + _content_loss + _tv_loss

        gradients = tape.gradient(
            total_loss, self.transformer_model.trainable_variables
        )
        self.optimizer.apply_gradients(
            zip(gradients, self.transformer_model.trainable_variables)
        )
        self.loss_metrics['total_loss'](total_loss)
        self.loss_metrics['style_loss'](_style_loss)
        self.loss_metrics['content_loss'](_content_loss)
        self.loss_metrics['total_variation_loss'](_tv_loss)

    def train(self, dataset, epochs: int, log_interval: int, notebook: bool):
        progress_bar = tqdm_notebook if notebook else tqdm
        for epoch in range(1, epochs + 1):
            for step, image in progress_bar(enumerate(dataset)):
                self.train_step(data=image)
                if step % log_interval == 0:
                    self._update_tensorboard(step=step)

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self.transformer_model.save_weights(
            filepath=filepath, overwrite=overwrite,
            save_format=save_format, options=options
        )

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        self.transformer_model.load_weights(
            filepath=filepath, by_name=by_name,
            skip_mismatch=skip_mismatch, options=options
        )
