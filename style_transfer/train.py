import os
import tensorflow as tf

from .dataloader import Dataloader
from .utils import gram_matrix, read_image
from .models import StyleContentModel, TransformerModel


class Trainer:

    def __init__(
            self, style_image_file: str, sample_content_image_file: str,
            experiment_name: str, wandb_api_key: str):
        self.style_image = read_image(style_image_file)
        self.sample_content_image = read_image(sample_content_image_file)
        self.experiment_name = experiment_name
        self.wandb_api_key = wandb_api_key
        self.feature_extractor = None
        self.transformer = None
        self.optimizer = None
        self.style_features = None
        self.gram_style = None
        self.checkpoint = None
        self.checkpoint_manager = None
        self.train_loss = None
        self.train_style_loss = None
        self.train_content_loss = None
        self.summary_writer = None
        self.dataset = None

    def _build_models(self):
        content_layers = ['block2_conv2']
        style_layers = [
            'block1_conv2', 'block2_conv2',
            'block3_conv3', 'block4_conv3'
        ]
        self.feature_extractor = StyleContentModel(
            style_layers=style_layers, content_layers=content_layers
        )
        self.transformer = TransformerModel()

    def _initialize_checkpoint(self):
        self.checkpoint = tf.train.Checkpoint(
            step=tf.Variable(1),
            optimizer=self.optimizer, transformer=self.transformer
        )
        log_dir = os.path.join('./logs', self.experiment_name)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, log_dir, max_to_keep=1)
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        if self.checkpoint_manager.latest_checkpoint:
            print(f"Restored from {self.checkpoint_manager.latest_checkpoint}")
        else:
            print("Initializing from scratch.")

    def _initialize_metrics(self):
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_style_loss = tf.keras.metrics.Mean(name="train_style_loss")
        self.train_content_loss = tf.keras.metrics.Mean(name="train_content_loss")

    def _initialize_summary_writer(self):
        log_dir = os.path.join('./logs', self.experiment_name)
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        with self.summary_writer.as_default():
            tf.summary.image(
                'Content Image',
                self.sample_content_image / 255.0, step=0
            )
            tf.summary.image(
                'Style Image',
                self.style_image / 255.0, step=0
            )

    def _build_dataset(self, image_size: int, batch_size: int):
        dataloader = Dataloader(image_size=image_size)
        self.dataset = dataloader.get_dataset(batch_size=batch_size)

    def build(self, learning_rate: float, image_size: int, batch_size: int):
        self._build_models()
        self.style_features, _ = self.feature_extractor(self.style_image)
        self.gram_style = [gram_matrix(x) for x in self.style_features]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self._initialize_checkpoint()
        self._initialize_metrics()
        self._initialize_summary_writer()
        self._build_dataset(image_size=image_size, batch_size=batch_size)
