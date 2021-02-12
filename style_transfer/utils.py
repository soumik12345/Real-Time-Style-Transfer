import tensorflow as tf


def read_img(image_file):
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
