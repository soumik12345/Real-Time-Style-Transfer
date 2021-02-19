from glob import glob

from style_transfer import Trainer
from style_transfer import Dataloader


dataloader = Dataloader(image_size=256)
dataset = dataloader.get_dataset(glob('./train2014/*.jph'), batch_size=16)

trainer = Trainer(
    experiment_name='experiment_1',
    wandb_api_key='696969696969696969699999696969696',
    style_image_file='324310.jpg', sample_content_image_file='5726.jpg',
    style_weight=10.0, content_weight=10.0, content_layers=['block2_conv2'],
    style_layers=['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']
)

trainer.compile(dataset=dataset, learning_rate=1e-3)

trainer.train(epochs=2, log_interval=500)
