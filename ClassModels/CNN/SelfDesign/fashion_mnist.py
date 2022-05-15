from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical 
# 使用cpu
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]='-1' 

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
(train_images , train_labels), (test_images, test_labels ) = fashion_mnist.load_data()


# # 映出圖片
# import cv2
# cv2.imshow('img',train_images[0])
# cv2.waitKey(0)
# # 映出答案
# print(train_labels[0])

# 檢查數據讀取
# plt.figure()
# plt.imshow(train_images[3],cmap="gray")
# plt.colorbar()
# plt.grid(True) #繪出網格線
# plt.show()

# 分類
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# normalize輸入的訓練集到0-1之間
train_images = train_images / 255.0
test_images = test_images / 255.0

# 新增到三維空間
X_train = np.expand_dims(train_images,axis=3)
X_test = np.expand_dims(test_images,axis=3)

print("X_train shape: ",X_train.shape)
print("X_test shape: ",X_test.shape)



# convert to one-hot-encoding(one hot vectors)
Y_train = to_categorical(train_labels)
Y_test = to_categorical(test_labels)

print("Y_train shape: ",Y_train.shape)
print("Y_test shape: ",Y_test.shape)


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 2)

model = Sequential()

#1. LAYER
model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))

#2. LAYER
model.add(Conv2D(filters = 48, kernel_size = (1,1), padding = 'Same'))
model.add(Conv2D(filters = 48, kernel_size = (3,3), padding = 'Same'))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

#3. LAYER

model.add(Conv2D(filters = 64, kernel_size = (1,1), padding = 'Same'))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same'))
model.add(BatchNormalization())
model.add(Activation("relu"))

#4. LAYER
model.add(Conv2D(filters = 64, kernel_size = (1,1), padding = 'Same'))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same'))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(MaxPool2D(pool_size=(2, 2)))

#FULLY CONNECTED LAYER
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Activation("relu"))

#OUTPUT LAYER
model.add(Dense(10, activation='softmax'))

model.summary()


optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 5 # for better result increase the epochs
batch_size = 127

datagen = ImageDataGenerator(
        featurewise_center=False,  # 以每一張feature map為單位將平均值設為0
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # 以每一張feature map為單位將數值除以其標準差(上述兩步驟就是我們常見的Standardization)
        samplewise_std_normalization=False,  #  將输入的每個樣本除以其自身的標準差。
        zca_whitening=False,  # dimesion reduction
        rotation_range=0.1,  # 隨機旋轉圖片
        zoom_range = 0.1, #  隨機縮放範圍
        width_shift_range=0.1,  #  水平平移，相對總寬度的比例
        height_shift_range=0.1,  # 垂直平移，相對總高度的比例
        horizontal_flip=False,  # 一半影象水平翻轉
        vertical_flip=False)  # 一半影象垂直翻轉

datagen.fit(x_train)

my_callbacks = [
    # validation loss 三個執行週期沒改善就停止訓練
    EarlyStopping(patience=3, monitor = 'val_accuracy'),
    # save the best weights
    # ModelCheckpoint(filepath="Component/CNN/mnist_model_weight.h5", verbose=1, save_best_only=True)
]

# # 載入最近的檢查點的權重
# model.load_weights(checkpoint_filepath)
# # 訓練 5 次
# model.fit(x_train_norm, y_train, epochs=5, validation_split=0.2, callbacks=model_checkpoint_callback)

# 降低資料佔據
history = model.fit(x=datagen.flow(x_train, y_train, batch_size=batch_size),shuffle=True,epochs=epochs, 
                              validation_data = datagen.flow(x_val, y_val, batch_size=batch_size),
                              steps_per_epoch=x_train.shape[0] // batch_size,
                              validation_steps = x_val.shape[0] // batch_size,
                              callbacks=my_callbacks) #we save the best weights with checkpointer


# 繪圖
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
plt.plot(history.history['acc'], color='green', label='Training Accuracy')
plt.plot(history.history['val_acc'], color='orange', label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()
