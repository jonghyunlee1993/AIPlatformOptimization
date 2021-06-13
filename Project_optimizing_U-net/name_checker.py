import tensorflow as tf
from src.model_keras import *

model = tf.keras.models.load_model(
    'weights/unet_best_keras', 
    custom_objects={'dice_coef_loss' : dice_coef_loss, 
                    'dice_coef' : dice_coef}
    )

print([node.op.name for node in model.inputs])