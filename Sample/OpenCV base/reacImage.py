import cv2
img = cv2.imread("./opencv/cat.jpg")
newImg = img[:100,200:300]
img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)

cv2.imshow("jpg", img)
cv2.imshow("newjpg", newImg)
cv2.waitKey(0)