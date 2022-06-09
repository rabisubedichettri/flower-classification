import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten,Dense,Dropout
from tensorflow.keras import Model
import os


from utils.config_loader import load_config,get_base_dic
from model.base_model import BaseModel


class AccuracyCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch,logs ={}):
        minimum_accuracy=self.config["custom_network"]["minimum_accuracy"]
        currnet_accuracy=logs.get('accuracy')
        if(currnet_accuracy > minimum_accuracy): 
            print(f'\n {currnet_accuracy}% acc reached')
            self.model.stop_training = True
        

class BridNet(BaseModel):
    def __init__(self, config):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        super().__init__(config)
        self.build()
        self.BASE_DIRECTORY=get_base_dic()

    def _preprocess(self):
        image_augment=self.config["train"]["image_augment"]
        self.train_datagen = ImageDataGenerator(rescale = 1./image_augment["rescale"],
                                        rotation_range =image_augment["rotation_range"],
                                        width_shift_range = image_augment["width_shift_range"],
                                        height_shift_range = image_augment["height_shift_range"],
                                        horizontal_flip = image_augment["horizontal_flip"],
                                        vertical_flip = image_augment["vertical_flip"],
                                        )
        self.test_datagen = ImageDataGenerator(rescale=1./self.config["test"]["image_augment"]["rescale"])
        self.valid_datagen = ImageDataGenerator(rescale=1./self.config["val"]["image_augment"]["rescale"])
        
    
    def data_loader(self):
        target_size = (244, 244)
        self.train_generator = self.train_datagen.flow_from_directory(
            self.config["train"]["location"],
            target_size=target_size,
            batch_size=self.config["train"]["batch_size"],
            class_mode='categorical')
        self.test_generator = self.test_datagen.flow_from_directory(
            self.config["test"]["location"],
            target_size=target_size,
            batch_size=self.config["test"]["batch_size"],
            class_mode='categorical')
        self.valid_generator = self.valid_datagen.flow_from_directory(
            self.config["val"]["location"],
            target_size=target_size,
            batch_size=self.config["val"]["batch_size"],
            class_mode='categorical')
    

    def build(self):
        base_last_layer=self.base_model.get_layer("mixed7")
        x=Flatten()(base_last_layer.output)
        x=Dense(1024,activation="relu")(x)
        x=Dropout(0.2)(x)
        y=Dense(self.config["custom_network"]["total_class"],activation="softmax")(x)
        self.model=Model(self.base_model.input,y)
        print("build model successfully")

    def compile(self):
        self.model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    
    def fit(self):
        # saved_model=self.config["saved_data"]
        # savedm_dir=os.path.join(get_base_dic(),saved_model["dir_name"])
        # if not os.path.exists(savedm_dir):
        #     os.mkdir(savedm_dir)
        # cp_dir_r=os.path.join(savedm_dir,saved_model["checkpoint"]["dir_name"])
        # if not os.path.exists(cp_dir_r):
        #     os.mkdir(cp_dir_r)
        # checkpoint_path = os.path.join(cp_dir_r,f'{saved_model["checkpoint"]["save"]}/cp.ckpt')
        

        # # Create a callback that saves the model's weights
        # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
        #                                          save_weights_only=True,
        #                                          verbose=1)
        self.model.fit(
            self.train_generator,
            epochs = self.config["custom_network"]["epochs"],
            validation_data=self.valid_datagen, 
            callbacks=[AccuracyCallbacks()]
        )



    def _model_summary(self):
        print(self.model.summary())

    def save(self):

        pass

    def _set_optimizer(self):
        pass

    def _set_callbacks(self):
        pass

    def train(self):
        self._preprocess()
        self.data_loader()
        self.build()
        # self._model_summary()
        self.compile()
        self.fit()


    def evaluate(self):
        pass


