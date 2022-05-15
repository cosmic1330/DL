import cv2
import numpy as np
img = np.zeros((600,600,3),np.uint8)
cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),3) # 來源,起點,終點,顏色,粗度
cv2.rectangle(img, (0,0), (300,300), (0,0,255), cv2.FILLED)
cv2.circle(img,(400,400),  30, (255,0,0), 2)
cv2.putText(img, "text", (300,300), cv2.FaceRecognizerSF_FR_COSINE, 1, (100,100,100), 2) # 來源,文字,座標,字體,大小,顏色,粗度
cv2.imshow('img', img)
cv2.waitKey(0)