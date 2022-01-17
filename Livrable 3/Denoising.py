import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import configuration
import os
import cv2
import numpy as np

data_dir = configuration.data_dir


class Denoising(tf.keras.Model):
    """docstring for Denoising"""

    def __init__(self):
        super(Denoising, self).__init__()
        self.inputs = None
        self.Model = None

    def build_autoencoder(self, height, width, depth, filters=(32, 64)):
        # initialize the input shape to be "channels last" along with
        # the channels dimension itself
        # channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1
        # define the input to the encoder
        inputs = layers.Input(shape=inputShape)
        x = inputs
        # loop over the number of filters
        for f in filters:
            # apply a CONV => MAX_POOLING
            x = layers.Conv2D(f, (3, 3), activation='relu', padding="same")(x)
            x = layers.MaxPooling2D((2, 2), padding='same')(x)

        # loop over our number of filters again, but this time in
        # reverse order
        for f in filters[::-1]:
            # apply a CONV => UP_SAMPLING
            x = layers.Conv2D(f, (3, 3), activation='relu', padding='same')(x)
            x = layers.UpSampling2D((2, 2))(x)
        # apply a single CONV_TRANSPOSE layer used to recover the
        # original depth of the image
        x = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        # our autoencoder is the encoder + decoder
        autoencoder = keras.models.Model(inputs, x,
                                         name="autoencoder")
        self.Model = autoencoder
        self.inputs = inputs

    def dataset(self):
        image_h = 304
        image_w = 304
        batch_s = 8
        validation_threshold = 0.2
        valid_images = [".jpg", ".gif", ".png"]
        clean_set = []
        noisy_set = []
        # we are loading only 100 image to speed up the process
        for file in os.listdir(data_dir + "/clean")[:100]:
            ext = os.path.splitext(file)[1]
            if ext.lower() not in valid_images or not os.path.isfile(os.path.join(data_dir + "/noise", file)):
                continue
            clean_img = cv2.imread(os.path.join(data_dir + "/clean", file))
            clean_img = cv2.resize(clean_img, (image_h, image_w))
            clean_img = clean_img.astype('float32') / 255.

            noisy_img = cv2.imread(os.path.join(data_dir + "/noise", file))
            noisy_img = cv2.resize(noisy_img, (image_h, image_w))
            noisy_img = noisy_img.astype('float32') / 255.
            clean_set.append(clean_img)
            noisy_set.append(noisy_img)
        clean_set = np.array(clean_set, dtype=object)
        noisy_set = np.array(noisy_set, dtype=object)
        (clean_set, noisy_set) = self.shuffle_in_unison(clean_set, noisy_set)

        split_index = int(noisy_set.shape[0] * (1 - validation_threshold))

        (clean_train_set, clean_test_set) = np.split(clean_set, [split_index])
        (noisy_train_set, noisy_test_set) = np.split(noisy_set, [split_index])

        # shuffle_in_unison can shuffle two array and keep index relations

    def shuffle_in_unison(a, b):
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b

    def compile_model(self):
        self.Model.compile(optimizer="adam", loss='binary_crossentropy')

    def load_model(self):
        self.Model.load_weights("./model_valid/autoencoder.h5")

    def summary_model(self):
        print(self.Model.summary())

    def predict(self, img, image_h, image_w):
        noisy_img = cv2.imread(img)
        noisy_img = cv2.resize(noisy_img, (image_h, image_w))
        noisy_img = noisy_img.astype('float32') / 255.

        decoded_img = self.Model.predict(tf.convert_to_tensor([noisy_img], dtype=tf.float32))
        return decoded_img
