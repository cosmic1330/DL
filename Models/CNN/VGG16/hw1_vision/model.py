
# windows
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop, Adadelta, Adam, SGD
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
import tensorflow as tf
tf.device("/device:GPU:0")
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# common

# 讀取資料－２
# training data path
trcloudy = './Component/weatherImage/train/cloudy'
trrain = './Component/weatherImage/train/rain'
trshine = './Component/weatherImage/train/shine'
trsunrise = './Component/weatherImage/train/sunrise'
# testing data path
testpath = './Component/weatherImage/test'

# 顯示資料筆數
print('train cloudy length: ', len(os.listdir(trcloudy)))
print('train rain length: ', len(os.listdir(trrain)))
print('train shine length: ', len(os.listdir(trshine)))
print('train sunrise length: ', len(os.listdir(trsunrise)))
print('test data length: ', len(os.listdir(testpath)))
print('\n')

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
            img = cv2.resize(img, (100,100))
            img = img[:, :, ::-1]/255
            data.append(img)
            label.append(index)
            fileNames.append(fileName)
            x += 1
    label = np.array(label)
    data = np.array(data)
    return data, label, fileNames


# 取得資料夾資料
trainData, trainLabel, trainFileNames = convertImageToNumpy(
    list=[trcloudy, trrain, trshine, trsunrise])
testData, testLabel, testFileNames = convertImageToNumpy(list=[testpath])

# on-hot label
trainLabel = to_categorical(trainLabel)


# define model
cnn = Sequential()
cnn.add(
    Conv2D(
        64, (3, 3),
        input_shape=(100,100, 3),
        kernel_regularizer=regularizers.l2(0.001),
        activation='relu', padding='same'))
cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

cnn.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
cnn.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
cnn.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

cnn.add(Flatten())
cnn.add(Dropout(0.25))
cnn.add(Dense(units=4,activation='softmax'))
# show the model structure
cnn.summary()

my_callbacks = [
    # validation loss 4個執行週期沒改善就停止訓練
    # EarlyStopping(patience=4, monitor='val_loss'),
    # save the best weights
    # ModelCheckpoint(
    #     filepath="Component/CNN/R37101426_楊竣宇_HW2/myModel_weight.h5", verbose=1,
    #     save_best_only=True)
    ReduceLROnPlateau(monitor = 'val_accuracy', patience = 3,verbose = 1, factor=0.5, min_lr = 0.00001)
]

# comiple model
batch_size = 64
epochs = 100
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# optimizer = RMSprop(learning_rate = 0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# optimizer = Adadelta(learning_rate=1.0, rho=0.95, epsilon=None, decay=0.0)
# optimizer = SGD(learning_rate=0.01, momentum=0.0, decay=0.0, nesterov=False)
cnn.compile(optimizer=optimizer, loss="categorical_crossentropy",
            metrics=['accuracy'])

# cross validation
acc_list = []
loss_list = []
counter = 1
for index in range(0, 1):
    # 打亂資料順序
    tr_data, tr_label = shuffle(trainData, trainLabel)
    # 準備此次的資料
    train_data = tr_data[0:786]
    val_data = tr_data[786:]
    train_label = tr_label[0:786]
    val_label = tr_label[786:]

    # # 映出圖片
    # cv2.imshow(f'train_label-{train_label[0]}',train_data[0])
    # cv2.imshow(f'train_label-{val_label[0]}',val_data[0])
    # cv2.waitKey(0)

    #  資料增生
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
    datagen.fit(train_data)

    # 使用此次訓練資料訓練
    history = cnn.fit(
        datagen.flow(train_data, train_label, batch_size=batch_size),
        epochs=epochs,
        validation_data=(val_data, val_label),
        steps_per_epoch=train_data.shape[0] // batch_size,
        validation_steps=val_data.shape[0] // batch_size,
        callbacks=my_callbacks)

    # 繪圖
    fit = plt.figure(figsize=(10, 10))
    plt.subplot(5, 3, counter)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('loss curve')
    plt.ylabel('loss')
    plt.legend()
    plt.subplot(5, 3, counter+1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('accuracy curve')
    plt.ylabel('accuracy')
    plt.legend()

    # 混淆舉證
    pre = cnn.predict(val_data)
    pre = np.argmax(pre, axis=1)
    cm = confusion_matrix(pre, np.argmax(val_label, axis=1))
    plt.subplot(5, 3, counter+2)
    plt.title('confusion matrix')
    sn.heatmap(cm, annot=True, cmap='OrRd', fmt='g')
    plt.xlabel('prediction')
    plt.ylabel('true label')

    #  圖檔位置
    counter+3

    #  計算loss
    score = cnn.evaluate(val_data,val_label)
    print("Test Loss:",score[0])
    print("Test Accuracy:",score[1])
plt.savefig('./Component/CNN/R37101426_楊竣宇_HW2/output.png',bbox_inches='tight')
plt.show()
cnn.save('./Component/CNN/R37101426_楊竣宇_HW2/myModel.h5') 
cnn.save_weights('./Component/CNN/R37101426_楊竣宇_HW2/myModel_weights.h5')


# 結果轉csv檔
prediction = cnn.predict(testData)
prediction = np.argmax(prediction,axis=1)
test_label = pd.DataFrame()
test_label['image_id']=testFileNames
test_label['labels']=prediction
test_label=test_label.sort_values(by='image_id')
test_label.to_csv('./Component/CNN/R37101426_楊竣宇_HW2/predict_label.csv',index=False)
