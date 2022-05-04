import numpy as np
import cv2
import random
img = np.empty((300,300,3), np.uint8) # 創建多微陣列
for row in range(300):
    for col in range(300):
        img[row][col] = [random.randint(0,255),100,3]

cv2.imshow('img',img)
cv2.waitKey(0)