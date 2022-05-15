from numpy import np
from keras.models import Sequential
from keras.layers import Dense, Activation


# 模型需要知道它所期望的輸入的尺寸
model = Sequential()
model.add(Dense(
    32,
    input_dim=784 # 或input_shape=(784,)
))
