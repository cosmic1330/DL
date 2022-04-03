import cv2
cap = cv2.VideoCapture("./opencv/videoplayback.mp4")
# cap = cv2.VideoCapture(0) # read camera

while True:
    ret, frame = cap.read() #ret:取得成功 frame:圖片
    if ret:
        frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
        cv2.imshow("video", frame)
    else:
        break

    if cv2.waitKey(10) == ord('q'):
        break

cv2.waitKey(0) 