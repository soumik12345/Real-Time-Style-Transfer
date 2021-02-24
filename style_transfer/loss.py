import tensorflow as tf


def style_loss(style_outputs, style_target):
    return tf.add_n([
        tf.reduce_mean((style_outputs[name] - style_target[name]) ** 2)
        for name in style_outputs.keys()
    ]) / len(style_outputs)


def content_loss(content_outputs, content_target):
    return tf.add_n([
        tf.reduce_mean((content_outputs[name] - content_target[name]) ** 2)
        for name in content_outputs.keys()
    ]) / len(content_outputs)


def total_variation_loss(img):
    return tf.reduce_mean(
        tf.square(img[:, :, 1:, :] - img[:, :, :-1, :])
    ) + tf.reduce_mean(
        tf.square(img[:, 1:, :, :] - img[:, :-1, :, :])
    )
