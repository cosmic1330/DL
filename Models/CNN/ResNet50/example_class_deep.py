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
# 數據下載
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True) 
# 方便的檔案路徑操作、簡易的檔案讀寫、並且幫你處理好跨作業系統問題
data_dir = pathlib.Path(data_dir) 

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory( 
  data_dir, # 将文件夹中的数据加载到tf.data.Dataset中
  validation_split=0.2, # 用於在沒有提供驗證集的時候，按一定比例從訓練集中取出一部分作為驗證集
  subset="training", # 標記為"training" 或 "validation"，如果在ImageDataGenerator中设置了validation_split
  seed=123, # 用於洗牌和轉換的可選隨機種子
  image_size=(img_height, img_width), # 從磁盤讀取圖像後調整圖像大小的大小
  batch_size=batch_size) # 數據批次的大小

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2, 
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# class_names ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
class_names = train_ds.class_names 

# 找到 prefetching 的最佳參數
AUTOTUNE = tf.data.AUTOTUNE 

# 轉換可以在內存或本地存儲中的緩存數據集。避免在每個 epoch 期間執行一些操作(如文件打開和數據讀取)。
train_ds = train_ds.cache()
# 洗牌
train_ds = train_ds.shuffle(1000) 
# 重疊訓練步驟的預處理和模型執行。當模型執行訓練步驟 s 時，讀取步驟 s+1 的數據。減少到訓練和提取數據所需的時間。
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE) 
val_ds = val_ds.cache()
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

#-----------------------------------------------------
# 資料增強
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),# RandomFlip("horizontal")：水平翻轉、
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),# RandomRotation(0.2)：旋轉 0.2 比例 
])

# 映出資料 9 筆訓練資料
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")

# 映出資料增強後 9 筆訓練資料
# for image, _ in train_ds.take(1): # 取出的是批量圖片向量，向量可用序號
#   plt.figure(figsize=(10, 10))
#   first_image = image[0]
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
#     plt.imshow(augmented_image[0] / 255)
#     plt.axis('off')

#-----------------------------------------------------
# 預先訓練好的模型 -- ResNet50
base_model = ResNet50(
  include_top=False, # 是否包含最後的全連接層 (fully-connected layer)
  weights='imagenet', # None: 權重隨機初始化、'imagenet': 載入預訓練權重
  input_shape=(180,180,3), # 當 include_top=False 時，可調整輸入圖片的尺寸（長寬需不小於 32）
  # input_tensor=None, # 使用 Keras tensor 作為模型的輸入層（layers.Input() 輸出的 tensor）
  # pooling=None, # 當 include_top=False 時，最後的輸出是否 pooling（可選 'avg' 或 'max'）
  # classes=1000 # 當 include_top=True 且 weights=None 時，最後輸出的類別數
)

base_model.trainable = False
inputs = tf.keras.Input(shape=(180,180,3))
# Image augmentation block
x = data_augmentation(inputs)
# 特徵縮放，每個特徵減掉該特徵的平均數
x = preprocess_input(x)
x = base_model(x,training=False)
x = GlobalAveragePooling2D()(x)
y = Dense(len(class_names), activation='softmax')(x) #final layer with softmax activation
model = Model(inputs=inputs, outputs=y, name="ResNet50")
# model.summary()
# tf.keras.utils.plot_model(model, show_shapes=True)

# 在新數據上訓練模型
model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=tf.metrics.SparseCategoricalAccuracy())
print("\n","train_ds",list(train_ds)[0][0].shape,"\n")
history = model.fit(train_ds, epochs=7, validation_data=val_ds)

#-----------------------------------------------------
# 畫圖
print(list(history.history))
plt.figure(figsize=(14,5))
plt.subplot(1, 2, 1)
plt.suptitle('Train Results', fontsize=10)
plt.xlabel("Number of Epochs")
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], color='b', label='Training Loss')
plt.plot(history.history['val_loss'], color='r', label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['sparse_categorical_accuracy'], color='green', label='Training Accuracy')
plt.plot(history.history['val_sparse_categorical_accuracy'], color='orange', label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()

#-----------------------------------------------------
# 預測
# image_batch,label_batch = val_ds.as_numpy_iterator().next()
# predictions = model.predict_on_batch(image_batch)


# for i in range(0,predictions.shape[0]):
#   print(np.argmax(predictions[i]))
#   prediction = np.argmax(predictions[i])
  
#   if (prediction != label_batch[i]):
#     plt.figure(figsize=(10, 10))
#     plt.imshow(prediction[i].numpy().astype("uint8"))
#     plt.title(class_names[label_batch[i]] + "-" + class_names[prediction])
#     plt.show()
    