import os
import wandb
from tqdm import tqdm
from typing import List
import tensorflow as tf

from .dataloader import Dataloader
from .utils import gram_matrix, read_image
from .loss import style_loss, content_loss
from .models import StyleContentModel, TransformerModel


class Trainer:

    def __init__(
            self, experiment_name: str, wandb_api_key: str,
            style_image_file: str, sample_content_image_file: str,
            style_weight: float, content_weight: float,
            content_layers: List[str], style_layers: List[str]):
        self.experiment_name = experiment_name
        self.wandb_api_key = wandb_api_key
        self.style_image = read_image(image_file=style_image_file)
        self.sample_content_image = read_image(image_file=sample_content_image_file)
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.dataset = None
        self.feature_extractor_model, self.transformer_model = None, None
        self.style_features, self.gram_style = None, None
        self.optimizer, self.summary_writer = None, None
        self.checkpoint, self.checkpoint_manager = None, None
        self.train_loss, self.train_content_loss, self.train_style_loss = None, None, None

    def _init_wandb(self):
        os.environ['WANDB_API_KEY'] = self.wandb_api_key
        wandb.init(
            project='real-time-style-transfer',
            name=self.experiment_name, sync_tensorboard=True
        )

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

    def _build_dataset(
            self, dataset_file_name: str, dataset_url: str,
            image_size: int, batch_size: int):
        dataloader = Dataloader(image_size=image_size)
        self.dataset = dataloader.get_dataset(
            dataset_file_name=dataset_file_name,
            dataset_url=dataset_url, batch_size=batch_size
        )

    def compile(
            self, dataset_file_name: str, dataset_url: str,
            image_size: int, batch_size: int, learning_rate: float):
        self._init_wandb()
        self._build_models()
        self._pre_compute_gram()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self._build_checkpoint_manager()
        self._initialize_metrics()
        self._initialize_summary_writer()
        self._build_dataset(
            dataset_file_name=dataset_file_name,
            dataset_url=dataset_url, image_size=image_size, batch_size=batch_size
        )

    @tf.function
    def _train_step(self, data):

        with tf.GradientTape() as tape:

            transformed_images = self.transformer_model(data)
            _, content_features = self.feature_extractor_model(data)
            (
                style_features_transformed,
                content_features_transformed
            ) = self.feature_extractor_model(transformed_images)
            total_style_loss = self.style_weight * style_loss(
                self.gram_style, style_features_transformed
            )
            total_content_loss = self.content_weight * content_loss(
                content_features, content_features_transformed
            )
            train_loss = total_style_loss + total_content_loss

        gradients = tape.gradient(
            train_loss, self.transformer_model.trainable_variables
        )
        self.optimizer.apply_gradients(
            zip(gradients, self.transformer_model.trainable_variables)
        )

        self.train_loss(train_loss)
        self.train_style_loss(total_style_loss)
        self.train_content_loss(total_content_loss)

    def _update_tensorboard(self, step: int):
        with self.summary_writer.as_default():
            tf.summary.scalar('loss', self.train_loss.result(), step=step)
            tf.summary.scalar('style_loss', self.train_style_loss.result(), step=step)
            tf.summary.scalar('content_loss', self.train_content_loss.result(), step=step)
            sample_styled_image = self.transformer_model(self.sample_content_image)
            tf.summary.image('Styled Image', sample_styled_image / 255.0, step=step)
        self.train_loss.reset_states()
        self.train_style_loss.reset_states()
        self.train_content_loss.reset_states()

    def train(self, epochs: int, log_interval: int):
        for epoch in range(1, epochs + 1):
            print('Epoch: ({}/{})'.format(epoch, epochs + 1))
            for data in tqdm(self.dataset):
                self._train_step(data=data)
                self.checkpoint.step.assign_add(1)
                step = int(self.checkpoint.step)
                if step % log_interval == 0:
                    self._update_tensorboard(step=step)
