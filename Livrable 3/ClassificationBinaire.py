import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import configuration
from tensorflow.keras.models import Sequential


class ClassificationBinaire(tf.keras.Model):

    def __init__(self, image_h, image_w, num_classes):
        super(ClassificationBinaire, self).__init__()
        self.validation_set = None
        self.train_set = None
        self.dataset(image_h, image_w)
        data_augmentation = keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip("horizontal",
                                                             input_shape=(image_h,
                                                                          image_w,
                                                                          3)),
                layers.experimental.preprocessing.RandomRotation(0.1),
                layers.experimental.preprocessing.RandomZoom(0.1)
            ]
        )
        self.model_multiple_layers = Sequential([
            data_augmentation,
            layers.Rescaling(1. / 255, input_shape=(image_h, image_w, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_classes)
        ])

    def dataset(self, image_h, image_w):
        batch_s = 32

        data_dir = configuration.data_dir
        self.train_set = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            labels="inferred",
            validation_split=0.2,
            subset="training",
            seed=42,
            batch_size=batch_s,
            image_size=(image_h, image_w),
        )
        # Le test_set
        self.validation_set = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            labels="inferred",
            validation_split=0.2,
            subset="validation",
            seed=42,
            batch_size=batch_s,
            image_size=(image_h, image_w),
        )
        class_names = self.train_set.class_names

    def caching(self):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.train_set = self.train_set.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.validation_set = self.validation_set.cache().prefetch(buffer_size=AUTOTUNE)

    def load_model(self):
        self.model_multiple_layers.load_weights("./model_valid/mulitple_layer_model.5.h5")

    # TO DO mettre le bon checkpoint dans L3

    def compile_model(self):
        self.model_multiple_layers.compile(optimizer='adam',
                                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                           metrics=['accuracy'])

    def summary_model(self):
        self.model_multiple_layers.summary()

    def predict(self, img):
        preds = self.model_multiple_layers.predict(img)
        return preds
