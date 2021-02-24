import tensorflow as tf
from matplotlib import pyplot as plt


def read_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_image(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = image[tf.newaxis, :]
    return image


def gram_matrix(input_tensor):
    result = tf.linalg.einsum(
        'bijc,bijd->bcd', input_tensor, input_tensor
    )
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(
        input_shape[1] * input_shape[2] * input_shape[3], tf.float32
    )
    return result / num_locations


def read_image(image_file: str):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_image(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = image[tf.newaxis, :]
    return image


def plot_result(style, content, stylized):
    fig = plt.figure(figsize=(12, 12))
    fig.add_subplot(1, 3, 1).set_title('Style Image')
    _ = plt.imshow(style)
    fig.add_subplot(1, 3, 2).set_title('Stylized Image')
    _ = plt.imshow(stylized)
    fig.add_subplot(1, 3, 3).set_title('Content Image')
    _ = plt.imshow(content)
    plt.show()
