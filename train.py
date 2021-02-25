import tensorflow as tf
from datetime import datetime

from style_transfer.utils import init_wandb
from style_transfer import StyleTransferModel, Dataloader


experiment_name = 'modified_transformer_vgg19_exp_1'

init_wandb(
    project_name='real-time-style-transfer',
    experiment_name=experiment_name,
    wandb_api_key='69696969696969696969696969696969696969'
)


dataset = Dataloader(image_size=256).get_dataset(image_files=[' '], batch_size=16)

model = StyleTransferModel(
    style_layers=['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'],
    content_layers=['block4_conv2'], style_image_file='324310.jpg', sample_content_image_file='5726.jpg',
    image_size=256, batch_size=16, experiment_name=experiment_name,
    style_weight=2e-3, content_weight=1.0, total_variation_weight=600.0
)

log_dir = "./logs/train/{}/".format(experiment_name) + datetime.now().strftime("%Y%m%d-%H%M%S")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), log_dir=log_dir)


model.train(dataset=dataset, epochs=2, log_interval=50, notebook=False)
