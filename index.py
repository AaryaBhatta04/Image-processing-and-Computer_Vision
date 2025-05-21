import cv2 as  cv
import numpy as np

def rescale_image(image,scale):
    width = int(image.shape[1]*scale)
    height = int(image.shape[0]*scale)
    
    dimension =(width,height)
    
    return cv.resize(image,dimension,interpolation=cv.INTER_AREA)
    


img = cv.imread('buildings.jpg')
img =rescale_image(img,0.25)

# cv.imshow("buildings",rescale_image(img,0.25))

# cv.waitKey(0)

#  CONVERT TO GRAYSCALE
# gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("x",gray_img)

# GUASSIAN BLUR
blur_image = cv.GaussianBlur(img,(5,5),cv.BORDER_DEFAULT)
# cv.imshow("blurred image",blur_image)

# CANNY EDGE DETECTION
canny = cv.Canny(blur_image,125,175)
cv.imshow('edges',canny)

resized = cv.resize(img,(500,500),interpolation = cv.INTER_AREA)
cv.imshow('resized',resized)
cv.waitKey(0)


