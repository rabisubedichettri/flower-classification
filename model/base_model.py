import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3

class BaseModel:
    def __init__(self,config):
        self.config=config
        base_network=self.config["base_network"]
        self.input_shape=(base_network["input_image_height"],base_network["input_image_width"],base_network["input_image_channel"])
        self.base_model = tf.keras.applications.InceptionV3( input_shape=self.input_shape,weights=base_network["weight"],include_top=base_network["include_top"])
        for layer in self.base_model.layers:
            layer.trainable = False