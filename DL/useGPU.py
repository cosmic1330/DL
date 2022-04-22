import tensorflow as tf
from tensorflow.python.client import device_lib

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

print(device_lib.list_local_devices())


# 使用CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"]='-1' 