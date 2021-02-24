import tensorflow as tf


class InferenceCallback(tf.keras.callbacks.Callback):

    def __init__(self, log_dir: str, style_image, sample_content_image):
        self.style_image = style_image
        self.sample_content_image = sample_content_image
        self.summary_writer = tf.summary.create_file_writer(logdir=log_dir)
        with self.summary_writer.as_default():
            tf.summary.image('Train/style_image', self.style_image / 255.0, step=0)
            tf.summary.image('Train/content_image', self.sample_content_image / 255.0, step=0)

    def on_batch_end(self, batch, logs=None):
        if batch % 50 == 0:
            sample_styled_image = self.transformer_model(self.sample_content_image)
            tf.summary.image('Train/styled_image', sample_styled_image / 255.0, step=batch)
