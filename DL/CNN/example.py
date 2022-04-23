from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(``
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])
for image, _ in train_ds.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')

base_model = ResNet50(include_top=False, weights='imagenet',input_shape=(180,180,3))
base_model.trainable = False
inputs = tf.keras.Input(shape=(180,180,3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x,training=False)
print(base_model.output.shape)
x = GlobalAveragePooling2D()(x)
y = Dense(5, activation='softmax')(x) #final layer with softmax activation
model = Model(inputs=inputs, outputs=y, name="ResNet50")
model.summary()

loss = tf.keras.losses.SparseCategoricalCrossentropy()
metrics = tf.metrics.SparseCategoricalAccuracy()
model.compile(optimizer='Adam', loss=loss, metrics=metrics)
len(model.trainable_variables)

history = model.fit(train_ds, epochs=7, validation_data=val_ds)

image_batch,label_batch = val_ds.as_numpy_iterator().next()
print(label_batch)
predictions = model.predict_on_batch(image_batch)

for i in range(0,predictions.shape[0]):
  print(np.argmax(predictions[i]))
  prediction = np.argmax(predictions[i])
  
  if (prediction != label_batch[i]):
    plt.figure(figsize=(10, 10))
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[label_batch[i]] + "-" + class_names[prediction])
    plt.show()