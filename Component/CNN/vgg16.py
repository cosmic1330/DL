

# mac 使用
import imp
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import RMSprop, Adadelta, Adam, SGD
from keras.losses import categorical_crossentropy, binary_crossentropy

# windows
# import os
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Flatten, Dropout, BatchNormalization
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras import regularizers
# import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.compat.v1.Session(config=config)

# common
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 讀取資料－１
# datagen = ImageDataGenerator(
#     featurewise_center=False,  # 以每一張feature map為單位將平均值設為0
#     samplewise_center=False,  # set each sample mean to 0
#     # 以每一張feature map為單位將數值除以其標準差(上述兩步驟就是我們常見的Standardization)
#     featurewise_std_normalization=False,
#     samplewise_std_normalization=False,  # 將输入的每個樣本除以其自身的標準差。
#     zca_whitening=False,  # dimesion reduction
#     rotation_range=0.1,  # 隨機旋轉圖片
#     zoom_range=0.1,  # 隨機縮放範圍
#     width_shift_range=0.1,  # 水平平移，相對總寬度的比例
#     height_shift_range=0.1,  # 垂直平移，相對總高度的比例
#     horizontal_flip=False,  # 一半影象水平翻轉
#     vertical_flip=False)  # 一半影象垂直翻轉
# train = datagen.flow_from_directory(
#     "./Component/weatherImage/train", target_size=(256, 256),shuffle=True,
#     batch_size=32, class_mode="categorical")

# test = datagen.flow_from_directory(
#     "./Component/weatherImage/test", target_size=(256, 256),shuffle=True,
#     batch_size=32, class_mode=".")

# 讀取資料－２
# training data path
trcloudy='./Component/weatherImage/train/cloudy'
trrain='./Component/weatherImage/train/rain'
trshine='./Component/weatherImage/train/shine'
trsunrise='./Component/weatherImage/train/sunrise'
# testing data path
testpath='./Component/weatherImage/test'
# 顯示資料筆數
print('train cloudy length: ',len(os.listdir(trcloudy)))
print('train rain length: ',len(os.listdir(trrain)))
print('train shine length: ',len(os.listdir(trshine)))
print('train sunrise length: ',len(os.listdir(trsunrise)))
print('test data length: ',len(os.listdir(testpath)))
print('\n')

# convert image data to numpy
def convertImageToNumpy(list):
    data=np.empty(shape=(1048,300,300,3))
    x=0
    label=[]
    fileNames=[]
    for index, url in enumerate(list):
        for i in range(len(os.listdir(url))):
            fileName=os.listdir(url)[i]
            img=cv2.imread(url+f"/{fileName}")
            img=cv2.resize(img,(300,300))
            img=img[:,:,::-1]/255
            data[x]=img
            label.append(index)
            fileNames.append(fileName)
            x+=1
    label=np.array(label)
    return data, label, fileNames

trainData,trainLabel, trainFileNames = convertImageToNumpy(list=[trcloudy,trrain,trshine,trsunrise]) # label=[0,1,2,3,4]
testData, testLabel, testFileNames = convertImageToNumpy(list=[testpath])

# on-hot label
trainLabel=to_categorical(trainLabel)
# 打亂資料順序
from sklearn.utils import shuffle
tr_data,tr_label=shuffle(trainData,trainLabel,random_state=0)
print("tr_data,tr_label len",len(tr_data),len(tr_label))
print('\n')

# train_validation data split
#val_data數量固定262個
train_data=trainData[0:786]
val_data=trainData[786:]
train_label=trainLabel[0:786]
val_label=trainLabel[786:]
print('train data size: ',train_data.shape)
print('validation data size: ',val_data.shape)
print('train label size: ',train_label.shape)
print('validation label size: ',val_label.shape)

# define model
cnn=Sequential() 
cnn.add(Conv2D(filters=64,kernel_size=(3,3), input_shape=(300,300,3),activation='relu',padding='same'))
cnn.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=(2,2),padding='same'))

cnn.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
cnn.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
cnn.add(BatchNormalization())
cnn.add(Dropout(0.25))
cnn.add(MaxPooling2D(pool_size=(2,2),padding='same'))

cnn.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
cnn.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
cnn.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
cnn.add(BatchNormalization())
cnn.add(Dropout(0.25))
cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

cnn.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
cnn.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
cnn.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
cnn.add(BatchNormalization())
cnn.add(Dropout(0.25))
cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

cnn.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
cnn.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
cnn.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
cnn.add(BatchNormalization())
cnn.add(Dropout(0.25))
cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

cnn.add(Flatten())
cnn.add(Dense(units=32,kernel_regularizer=regularizers.l2(l=0.001),activation='relu'))
cnn.add(Dense(units=16,kernel_regularizer=regularizers.l2(l=0.001),activation='relu'))
cnn.add(Dense(units=4,activation='softmax'))
# show the model structure
cnn.summary()

from keras.preprocessing.image import ImageDataGenerator ##Augmentation

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
datagen.fit(train_data)

my_callbacks = [
    # validation loss 三個執行週期沒改善就停止訓練
    EarlyStopping(patience=3, monitor = 'val_accuracy'),
    # save the best weights
    ModelCheckpoint(filepath="Component/CNN/vgg16_model_weight.h5", verbose=1, save_best_only=True)
]


# comiple model
# optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False )
# optimizer = RMSprop(lr = 0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
# optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

cnn.compile(optimizer="adam",loss=binary_crossentropy,metrics=['accuracy'])
history = cnn.fit_generator(datagen.flow(train_data, train_label, batch_size=32) , epochs=20, validation_data = (val_data, val_label),callbacks = my_callbacks)
# history = cnn.fit(train_data, train_label, batch_size=32, epochs=20, validation_data = (val_data, val_label), callbacks=my_callbacks)
print("++ finish training ++")

# 繪圖
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.title('loss curve')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.plot(history.history['acc'],label='accuracy')
plt.plot(history.history['val_acc'],label='val_accuracy')
plt.title('accuracy curve')
plt.ylabel('accuracy')
plt.legend()
plt.show()

# 混淆舉證
import seaborn as sn
from sklearn.metrics import confusion_matrix
pre=cnn.predict(val_data)
pre=np.argmax(pre,axis=1)
cm=confusion_matrix(pre,np.argmax(val_label,axis=1))
fit=plt.figure(figsize=(8,6))
plt.title('confusion matrix')
sn.heatmap(cm,annot=True,cmap='OrRd',fmt='g')
plt.xlabel('prediction')
plt.ylabel('true label')
plt.show()

# 結果轉csv檔
import pandas as pd
prediction=cnn.predict(testData)
prediction=np.argmax(prediction,axis=1)
prediction
test_label=pd.DataFrame()
test_label['image_id']=testFileNames
test_label['labels']=prediction
test_label=test_label.sort_values(by='image_id')
test_label.to_csv('/content/drive/MyDrive/weather_image/predict_label.csv',index=False)