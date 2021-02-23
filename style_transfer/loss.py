import tensorflow as tf

from .utils import gram_matrix


def style_loss(gram_style, style_features_transformed):
    return tf.add_n(
        [
            tf.reduce_mean((gram_matrix(sf_transformed) - gm) ** 2)
            for sf_transformed, gm in zip(
                style_features_transformed, gram_style
            )
        ]
    )


def content_loss(content_features, content_features_transformed):
    return tf.add_n(
        [
            tf.reduce_mean((cf_transformed - cf) ** 2)
            for cf_transformed, cf in zip(
                content_features_transformed, content_features
            )
        ]
    )


def total_variation_loss(image):
    return tf.reduce_mean(
        tf.square(image[:, :, 1:, :] - image[:, :, :-1, :])
    ) + tf.reduce_mean(
        tf.square(image[:, 1:, :, :] - image[:, :-1, :, :])
    )
