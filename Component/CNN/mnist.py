
# mac 使用
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

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

print(train_images.shape)

X_train = np.expand_dims(train_images,axis=3)
X_test = np.expand_dims(test_images,axis=3)

print("X_train shape: ",X_train.shape)
print("X_test shape: ",X_test.shape)

from keras.utils.np_utils import to_categorical 

# convert to one-hot-encoding(one hot vectors)
Y_train = to_categorical(train_labels, num_classes = 10)
# convert to one-hot-encoding(one hot vectors)
Y_test = to_categorical(test_labels, num_classes = 10)

print(Y_train.shape)
print(Y_test.shape)