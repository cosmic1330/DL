from numpy import np

# Sequential
from keras.models import Sequential
from keras.layers import Dense, Activation

# 堆疊模型的方法
model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
