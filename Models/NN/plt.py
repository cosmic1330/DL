
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential


from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
image_size = x_train.shape[1]
input_size = image_size * image_size
x_test = np.reshape(x_test, [-1, input_size])
x_test = x_test.astype('float32') / 255


model = Sequential()
model = load_model("./DL/main.h5")
prediction=np.argmax(model.predict(x_test), axis=-1)
print(prediction.shape)
# model.summary()
# history=np.load('./DL/main_history.npy',allow_pickle='TRUE').item()

# print(history)
# plt.plot(history['loss'])
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.title('Loss')
# plt.show()

# plt.plot(history['accuracy'])
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.title('Accuracy')
# plt.show()
