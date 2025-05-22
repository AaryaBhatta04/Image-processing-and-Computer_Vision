import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# note -> does_not work as well as opencv builtin function, and is slow also

def Canny_detector(img, weak_threshold=None, strong_threshold=None):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5, 5), cv.BORDER_DEFAULT)
    gradient_x = cv.Sobel(np.float32(img), cv.CV_64F, 1, 0, 3)
    gradient_y = cv.Sobel(np.float32(img), cv.CV_64F, 0, 1, 3)
    mag, ang = cv.cartToPolar(gradient_x, gradient_y, angleInDegrees=True)
    mag_normalized = np.uint8(255 * mag / np.max(mag))
    nms = np.zeros_like(mag_normalized, dtype=np.uint8)
    ang = np.mod(ang, 180) 
    
    for i in range(1, mag.shape[0] - 1):
        for j in range(1, mag.shape[1] - 1):
            q, r = 255, 255
            if (0 <= ang[i, j] < 22.5) or (157.5 <= ang[i, j] <= 180):
                q = mag[i, j + 1]
                r = mag[i, j - 1]
            elif 22.5 <= ang[i, j] < 67.5:
                q = mag[i + 1, j - 1]
                r = mag[i - 1, j + 1]
            elif 67.5 <= ang[i, j] < 112.5:
                q = mag[i + 1, j]
                r = mag[i - 1, j]
            elif 112.5 <= ang[i, j] < 157.5:
                q = mag[i - 1, j - 1]
                r = mag[i + 1, j + 1]
            
            
            if (mag[i, j] >= q) and (mag[i, j] >= r):
                nms[i, j] = mag_normalized[i, j]
            else:
                nms[i, j] = 0
    
    
    if weak_threshold is None:
        weak_threshold = 0.1 * np.max(nms)
    if strong_threshold is None:
        strong_threshold = 0.2 * np.max(nms)
    
    
    strong_edges = (nms > strong_threshold)
    weak_edges = (nms >= weak_threshold) & (nms <= strong_threshold)
    edges = np.zeros_like(nms, dtype=np.uint8)
    edges[strong_edges] = 255  
    
    
    for i in range(1, edges.shape[0] - 1):
        for j in range(1, edges.shape[1] - 1):
            if weak_edges[i, j]:
                if np.any(strong_edges[i-1:i+2, j-1:j+2]):
                    edges[i, j] = 255 
    
    return edges


img = cv.imread("buildings.jpg") 
edges = Canny_detector(img)
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)), plt.title("Original")
plt.subplot(122), plt.imshow(edges, cmap='gray'), plt.title("Canny Edges")
plt.show()