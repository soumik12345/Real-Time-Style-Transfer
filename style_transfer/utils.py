import os
import wandb
import tensorflow as tf


def init_wandb(project_name: str, experiment_name: str, wandb_api_key: str):
    os.environ['WANDB_API_KEY'] = wandb_api_key
    wandb.init(
        project=project_name,
        name=experiment_name, sync_tensorboard=True
    )


def read_image(image_file, image_size):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_image(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [image_size, image_size])
    image = image[tf.newaxis, :]
    return image


def gram_matrix(features, normalize: bool = True):
    batch_size, height, width, filters = features.shape
    features = tf.reshape(features, (batch_size, height * width, filters))
    tran_f = tf.transpose(features, perm=[0, 2, 1])
    gram = tf.matmul(tran_f, features)
    if normalize:
        gram /= tf.cast(height * width, tf.float32)
    return gram
