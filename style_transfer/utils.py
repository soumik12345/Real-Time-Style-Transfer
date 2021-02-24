import tensorflow as tf


def gram_matrix(features, normalize: bool = True):
    batch_size, height, width, filters = features.shape
    features = tf.reshape(features, (batch_size, height*width, filters))
    tran_f = tf.transpose(features, perm=[0, 2, 1])
    gram = tf.matmul(tran_f, features)
    if normalize:
        gram /= tf.cast(height*width, tf.float32)
    return gram
