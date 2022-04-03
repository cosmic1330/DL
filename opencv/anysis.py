# 偵測顏色
import cv2
import numpy as np
img = cv2.imread("opencv/cat.jpg")
img = cv2.resize(img, (0,0), fx=2,fy=2) #放大兩倍

# hsv比rgb更容易過濾顏色
hsv= cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #HSV 色調 飽和度 亮度

# empty 控制條改變後呼叫的函示
def empty(v):
    pass

cv2.namedWindow('TrackBar')
cv2.resizeWindow('TrackBar', 640, 320)
cv2.createTrackbar('Hue Min', 'TrackBar', 0, 179, empty)
cv2.createTrackbar('Hue Max', 'TrackBar', 179, 179, empty)
cv2.createTrackbar('Sat Min', 'TrackBar', 0, 255, empty)
cv2.createTrackbar('Sat Max', 'TrackBar', 255, 255, empty)
cv2.createTrackbar('Val Min', 'TrackBar', 0, 255, empty)
cv2.createTrackbar('Val Max', 'TrackBar', 255, 255, empty)

while True:
    h_min=cv2.getTrackbarPos('Hue Min', 'TrackBar')
    h_max=cv2.getTrackbarPos('Hue Max', 'TrackBar')
    s_min=cv2.getTrackbarPos('Sat Min', 'TrackBar')
    s_max=cv2.getTrackbarPos('Sat Max', 'TrackBar')
    v_min=cv2.getTrackbarPos('Val Min', 'TrackBar')
    v_max=cv2.getTrackbarPos('Val Max', 'TrackBar')
    lower=np.array([h_min,s_min,v_min])
    upper=np.array([h_max,s_max,v_max])
    mask = cv2.inRange(hsv,lower,upper)
    result = cv2.bitwise_and(img, img, mask=mask) #幫我們把兩張圖片每一個bit做and(&&)運算 只有在a和b都是1的時候才是1

    cv2.imshow('mask', mask)
    cv2.imshow('img', img)
    cv2.imshow('result', result)
    cv2.waitKey(1)

    