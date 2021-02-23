from PIL import Image
import tensorflow as tf
from .utils import read_image
from .models import TransformerModel


class Inferer:

    def __init__(self):
        self.model = None

    def compile(self, weights_path: str):
        self.model = TransformerModel()
        self.model.load_weights(weights_path)

    def infer(self, image_file: str, output_path=None):
        image = read_image(image_file)
        stylized_image = self.model(image)
        stylized_image = tf.cast(
            tf.squeeze(stylized_image), tf.uint8
        ).numpy()
        stylized_image = Image.fromarray(stylized_image, mode='RGB')
        if output_path is not None:
            stylized_image.save(output_path)
        else:
            return stylized_image
