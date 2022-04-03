import cv2
img=cv2.imread("opencv/cat.jpg")
print(img.shape) # (224,225,3)

# [
#     [[[0,255,0],[255,0,0],[0,0,255]],[],[],...225],
#     [],
#     [],
#     ...224
# ]