
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


path = './opencv/cat.jpg'
img= cv2.imread(path)
train_data = np.array([img,img,img,img,img,img])

datagen = ImageDataGenerator(
    featurewise_center=False,  # 以每一張feature map為單位將平均值設為0
    samplewise_center=False,  # set each sample mean to 0
    # 以每一張feature map為單位將數值除以其標準差(上述兩步驟就是我們常見的Standardization)
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,  # 將输入的每個樣本除以其自身的標準差。
    zca_whitening=False,  # dimesion reduction
    rotation_range=0.1,  # 隨機旋轉圖片
    zoom_range=0.1,  # 隨機縮放範圍
    width_shift_range=0.1,  # 水平平移，相對總寬度的比例
    height_shift_range=0.1,  # 垂直平移，相對總高度的比例
    horizontal_flip=False,  # 一半影象水平翻轉
    vertical_flip=False)  # 一半影象垂直翻轉

# datagen.fit(train_data)
gener = datagen.flow(train_data,[1,0,0,0,0,0],batch_size=4,save_to_dir="./Tensorflow/imgs",save_prefix="trans_")
print("經過處理的 len:",gener[0][0].shape,gener[0][1].shape,"\n未經過處理的 len",gener[1][0].shape,gener[0][1].shape,)

# newImg = gener.next()
# print(newImg[0].shape, newImg[1].shape)