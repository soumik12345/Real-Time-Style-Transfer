from glob import glob
import tensorflow as tf
import tensorflow_datasets as tfds


class TFDSDataloader:

    def __init__(self, image_size: int):
        self.image_size = image_size

    def _map_function(self, features):
        image = features["image"]
        image = tf.image.resize(
            image, size=(self.image_size, self.image_size)
        )
        image = tf.cast(image, tf.float32)
        return image

    def get_dataset(self, dataset_name: str = 'coco/2014', batch_size: int = 16):
        dataset = tfds.load(dataset_name, split='train')
        dataset = dataset.map(
            self._map_function,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


class Dataloader:

    def __init__(self, image_size: int):
        self.image_size = image_size

    def _map_function(self, image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(
            image, size=(self.image_size, self.image_size)
        )
        image = tf.cast(image, tf.float32)
        return image

    def get_dataset(self, dataset_file_name: str, dataset_url: str, batch_size: int = 16):
        dataset_location = tf.keras.utils.get_file(
            fname=dataset_file_name, origin=dataset_url, extract=True
        )
        dataset_location = dataset_location.split('.')[0]
        image_files = glob(dataset_location + '*.jpg')
        dataset = tf.data.Dataset.from_tensor_slices(image_files)
        dataset = dataset.map(
            self._map_function,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
