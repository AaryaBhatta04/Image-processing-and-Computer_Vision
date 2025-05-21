import cv2 as cv
import numpy as np

def rescaler(image,scale):
    width = int(image.shape[1]*scale)
    height = int(image.shape[0]*scale)
    
    dimensions = (width,height)
    
    return cv.resize(image,dimensions,interpolation = cv.INTER_AREA)

img = cv.imread('buildings.jpg')
img = rescaler(img,0.20)
cv.imshow('image',img)

def translate(img,x,y):
    transMat = np.float32([[1,0,x],[0,1,y]]) 
    dimensions = (img.shape[1],img.shape[0]) 
    
    return cv.warpAffine(img,transMat,dimensions)

translated = translate(img,100,-100)

cv.imshow('shifted',translated)

cv.waitKey(0)