class ClassificationBinaire(tf.keras.Model):

  def __init__(self):
    super(ClassificationBinaire, self).__init__()
    self.train_set = undefined
    self.validation_set = undefined
    self.model_multiple_layers = Sequential([
      data_augmentation,
      layers.Rescaling(1./255, input_shape=(image_h, image_w, 3)),
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



def dataset():
  image_h = 180
  image_w = 180
  batch_s = 32

  data_dir = configuration.data_dir
  self.train_set = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels = "inferred",
    validation_split= 0.2,
    subset = "training",
    seed=42,
    batch_size=batch_s,
    image_size=(image_h, image_w),
  )
  # Le test_set
  self.validation_set = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels = "inferred",
    validation_split= 0.2,
    subset = "validation",
    seed=42,
    batch_size=batch_s,
    image_size=(image_h, image_w),
  )
  class_names = train_set.class_names

def caching():
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  self.train_set = self.train_set.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
  self.validation_set = self.validation_set.cache().prefetch(buffer_size=AUTOTUNE)


def load_model():
  self.model_multiple_layers.load_weights("./training/mulitple_layer_model.5.h5")
# TO DO mettre le bon checkpoint dans L3


def compile_model():
  self.model_multiple_layers.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

def summary_model():
  self.model_multiple_layers.summary()


def predict(img):
  preds = self.model_multiple_layers.predict(img)
  return preds



