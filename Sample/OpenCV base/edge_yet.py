# 輪廓檢測 和邊緣檢測不同
import cv2
import numpy as np
img = cv2.imread("./opencv/cat.jpg")

# imgContour=img.copy()
imgContour=np.zeros((img.shape[1],img.shape[0]))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #因為檢測不需要顏色
canny = cv2.Canny(img,100,200) #來源,最低門檻值,最高門檻值
contours,hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 把輪廓畫在圖片上
for cnt in contours:
    # print(cnt) # 映出輪廓點
    cv2.drawContours(imgContour, cnt, -1, (255,255,255), 1)  #劃出輪廓 畫的圖片,要畫得輪廓點, 要畫第幾個輪廓, 顏色,粗度
    # print(cv2.contourArea(cnt)) #映出輪廓面積
    peri=cv2.arcLength(cnt, True)
    # print(peri) #應出輪廓長度 (0是噪點)
    vertices=cv2.approxPolyDP(cnt, peri*0.02, True)
    print(len(vertices)) #映出頂點看他是幾邊形

    
cv2.imshow('imgContour',imgContour)
cv2.waitKey(0)