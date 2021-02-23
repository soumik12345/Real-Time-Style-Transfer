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

    def infer(self, image_file: str):
        image = read_image(image_file)
        stylized_image = self.model(image)
        stylized_image = tf.cast(
            tf.squeeze(stylized_image), tf.uint8
        ).numpy()
        return Image.fromarray(stylized_image, mode='RGB')
