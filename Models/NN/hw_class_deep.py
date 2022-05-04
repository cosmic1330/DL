# mac 使用
# import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Dropout
# from keras.utils import to_categorical
# from keras.datasets import mnist

# windows 使用
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 分為訓練資料和測試資料(圖片,答案)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# # 映出圖片
# import cv2
# cv2.imshow('img',x_train[0])
# cv2.waitKey(0)
# # 映出答案
# print(y_train[0])

# 去除重複的答案，取得答案Class的數量
num_labels = len(np.unique(y_train))
y_test_origin = y_test
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

image_size = x_train.shape[1]
input_size = image_size * image_size
# 先將原本28*28的2維數字影像，以reshape()轉換成1維向量，再以astype()轉換為float
x_train = np.reshape(x_train, [-1, input_size]).astype('float32') 
x_train = x_train / 255 # normal
x_test = np.reshape(x_test, [-1, input_size]).astype('float32')
x_test = x_test / 255

epochs = 20
batch_size = 129

model = Sequential()
# 第一層
model.add(Dense(256, input_dim=input_size))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 隱藏層
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 輸出層
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()

model.compile(
              loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

myModel=model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

loss, accuracy = model.evaluate(x_test,
                        y_test,
                        batch_size=batch_size,
                        verbose=0)
print("\nTest accuracy: %.1f%%" % (100.0 * accuracy))

# 儲存model和結果
model.save("./DL/mnist/mnist.h5",myModel.history)
np.save('./DL/mnist/mnist_history.npy',myModel.history)


# 預測
print("預測機率:",model.predict(x_test))
print("預測答案:",np.argmax(model.predict(x_test), axis=-1))

# 預測圖形
# def plot_images_labels(images,labels,prediction,idx,num=10):
#     fig=plt.gcf()
#     fig.set_size_inches(12,14)
#     if num>25:
#         num=15
#     for i in range(0,num):
#         ax=plt.subplot(5,5,1+i)
#         ax.imshow(np.reshape(images[idx],(28,28)), cmap='binary')
#         # title="label=" +str(labels[idx])
#         if len(prediction)>0:
#             title=",predict="+str(prediction[idx])
#         ax.set_title(title,fontsize=10)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         idx=idx+1
#     plt.show()
# plot_images_labels(x_test,y_test,prediction,idx=300)

# 混淆矩陣
# prediction=np.argmax(model.predict(x_test), axis=-1)
# res = pd.crosstab(y_test_origin,prediction,rownames=['label'],colnames=['predict'])
# print(res)


print(myModel.history)
# plt.figure(figsize=(20,40))
# plt.subplot(1,2,1)
# plt.plot(myModel.history['loss'])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.title('Loss')
# plt.show()

# plt.subplot(1,2,2)
# plt.plot(myModel.history['acc'])
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.title('Accuracy')
# plt.show()