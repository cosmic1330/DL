
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import os
import cv2

# convert image data to numpy
def convertImageToNumpy(list):
    data = []
    x = 0
    label = []
    fileNames = []
    for index, url in enumerate(list):
        for i in range(len(os.listdir(url))):
            fileName = os.listdir(url)[i]
            img = cv2.imread(url+f"/{fileName}")
            img = cv2.resize(img, (64, 64))
            img = img[:, :, ::-1]/255
            data.append(img)
            label.append(index)
            fileNames.append(fileName)
            x += 1
    label = np.array(label)
    data = np.array(data)
    return data, label, fileNames

# training data path
trcloudy = './Component/weatherImage/train/cloudy'
trrain = './Component/weatherImage/train/rain'
trshine = './Component/weatherImage/train/shine'
trsunrise = './Component/weatherImage/train/sunrise'
# testing data path
testpath = './Component/weatherImage/test'

# 取得資料夾資料
trainData, trainLabel, trainFileNames = convertImageToNumpy(
    list=[trcloudy, trrain, trshine, trsunrise])
testData, testLabel, testFileNames = convertImageToNumpy(list=[testpath])

# on-hot label
trainLabel = tf.keras.utils.to_categorical(trainLabel)
# 打亂資料順序
tr_data, tr_label = shuffle(trainData, trainLabel)

# 準備此次的資料
train_data = tr_data[0:786]
val_data = tr_data[786:]
train_label = tr_label[0:786]
val_label = tr_label[786:]


# --------------------------------------------------------------------------------
# 從 HDF5 檔案中載入模型
model = tf.contrib.keras.models.load_model('Component/CNN/vgg16.h5')

# 驗證模型
score = model.evaluate(x_test, y_test, verbose=0)

# 輸出結果
print('Test loss:', score[0])
print('Test accuracy:', score[1])
