import os
from typing import List
import tensorflow as tf
from datetime import datetime

from .dataloader import Dataloader
from .loss import style_loss, content_loss, gram_matrix
from .utils import gram_matrix, read_image
from .models import StyleContentModel, TransformerModel


class Trainer:

    def __init__(
            self, experiment_name: str,
            style_image_file: str, sample_content_image_file: str,
            content_layers: List[str], style_layers: List[str]):
        self.experiment_name = experiment_name
        self.style_image = read_image(image_file=style_image_file)
        self.sample_content_image = read_image(image_file=sample_content_image_file)
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.feature_extractor_model, self.transformer_model = None, None
        self.style_features, self.gram_style = None, None
        self.optimizer, self.summary_writer = None, None
        self.checkpoint, self.checkpoint_manager = None, None
        self.train_loss, self.train_content_loss, self.train_style_loss = None, None, None

    def _build_models(self):
        self.feature_extractor_model = StyleContentModel(
            style_layers=self.style_layers, content_layers=self.content_layers
        )
        self.transformer_model = TransformerModel()

    def _pre_compute_gram(self):
        self.style_features, _ = self.feature_extractor_model(self.style_image)
        self.gram_style = [gram_matrix(x) for x in self.style_features]

    def _build_checkpoint_manager(self):
        self.checkpoint = tf.train.Checkpoint(
            step=tf.Variable(1),
            optimizer=self.optimizer,
            transformer=self.transformer_model
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            './logs/checkpoints/{}'.format(
                self.experiment_name
            ), max_to_keep=1
        )
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        if self.checkpoint_manager.latest_checkpoint:
            print('Restored Checkpoint from {}'.format(
                self.checkpoint_manager.latest_checkpoint
            ))
        else:
            print('Training from scratch....')

    def _initialize_metrics(self):
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_style_loss = tf.keras.metrics.Mean(name='train_style_loss')
        self.train_content_loss = tf.keras.metrics.Mean(name='train_content_loss')

    def _initialize_summary_writer(self):
        self.summary_writer = tf.summary.create_file_writer(
            './logs/train/{}'.format(self.experiment_name)
        )
        with self.summary_writer.as_default():
            tf.summary.image('Style Image', self.style_image / 255.0, step=0)
            tf.summary.image('Content Image', self.sample_content_image / 255.0, step=0)

    def compile(self, learning_rate: float):
        self._build_models()
        self._pre_compute_gram()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self._build_checkpoint_manager()
        self._initialize_metrics()
        self._initialize_summary_writer()
