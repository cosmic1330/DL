import matplotlib.pyplot as plt
import os
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

dirpath = './images/stanfor_dogs/'
os.listdir(dirpath)

imgSize = (224, 224)
imgShape = (224, 224, 3)
batchSize = 128
epoch = 40

# generator
trainDataGenerator = ImageDataGenerator(rescale=1/255,
                                        rotation_range=30,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        fill_mode="nearest",
                                        validation_split=0.2)
valDataGenerator = ImageDataGenerator(rescale=1/255, validation_split=0.2)
testDataGenerator = ImageDataGenerator(rescale=1/255)

#  資料生成
trainBatch = trainDataGenerator.flow_from_directory(
    directory=dirpath+'train',
    target_size=imgSize,
    class_mode='categorical',
    shuffle=True,
    batch_size=batchSize,
    seed=200,
    subset='training',
)
valBatch = valDataGenerator.flow_from_directory(
    directory=dirpath+'train',
    target_size=imgSize,
    class_mode='categorical',
    shuffle=True,
    batch_size=batchSize,
    seed=200,
    subset='validation',
)
testBatch = testDataGenerator.flow_from_directory(
    dirpath,
    class_mode=None,
    batch_size=1,
    shuffle=False,
    target_size=imgSize,
    classes=['test'],
)

# model
base_model = InceptionResNetV2(
    include_top=False,
    weights='imagenet',
    input_tensor = None,
    input_shape=imgShape,
)


# freeze the weight
for layer in base_model.layers:
    layer.trainable = True

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
output_layer = layers.Dense(120, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output_layer)
# model.summary()


# compile
model.compile(
    # optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3, decay=1e-3,momentum=0.9),
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-4),
    loss='categorical_crossentropy', metrics=['accuracy'])

# callback
model_checkpoint_callback = ModelCheckpoint(
    filepath="./Models/CNN/ResNet50/hw3_vision/checkpoint/main.h5",
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max', verbose=1,
    save_best_only=True)
model_earlyStopping_callback = EarlyStopping(
    monitor="val_accuracy", patience=5, mode="max", verbose=1,)

# continue
model.load_weights("./Models/CNN/ResNet50/hw3_vision/checkpoint/main.h5", by_name = True)

# fit
history = model.fit(
    trainBatch,
    steps_per_epoch=trainBatch.samples // batchSize,
    validation_data=valBatch,
    validation_steps=valBatch.samples // batchSize,
    epochs=epoch,
    verbose=1,
    callbacks=[model_checkpoint_callback, model_earlyStopping_callback]
)

# load_weight
model.load_weights("./Models/CNN/ResNet50/hw3_vision/checkpoint/main.h5")


# predict
prediction = model.predict(testBatch)
prediction_label = prediction.argmax(axis=1)
filename = testBatch.filenames
for i in range(len(filename)):
    filename[i] = filename[i].replace('test/', '')
outputdf = pd.DataFrame()
outputdf['Name'] = filename
outputdf['Label'] = prediction_label
outputdf.to_csv('./Models/CNN/ResNet50/hw3_vision/prediction.csv', index=False)

# show image
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('loss curve')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('accuracy curve')
plt.ylabel('accuracy')
plt.legend()
plt.show()
