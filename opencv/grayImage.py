import cv2
import numpy as np #使用陣列時需要
kernal = np.ones((10,10),np.uint8)
img= cv2.imread("opencv/cat.jpg")
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(img, (5,5), 0.5) #src,kernal,標準差
canny = cv2.Canny(img, 100 , 150) #門檻越低 邊緣越多
dialate = cv2.dilate(img, kernal, iterations = 1) #膨脹的效果(邊緣變粗) 次數
erode = cv2.erode(dialate,kernal, iterations = 1 )

# cv2.imshow('img',img)
# cv2.imshow('gray',gray)
# cv2.imshow('blur',blur)
# cv2.imshow('canny',canny)
cv2.imshow('dialate',dialate)
cv2.imshow('erode',erode)
cv2.waitKey(0)