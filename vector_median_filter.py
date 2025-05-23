import cv2
import numpy as np

# I am using mask size = 3

def vector_median_filter(img, mask_size):
    height, width, channels = img.shape
    pad = mask_size // 2
    padded_img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    filtered_img = np.zeros_like(img)
    
    for i in range(height):
        for j in range(width):
            neighborhood = padded_img[i:i+mask_size, j:j+mask_size]
            vectors = neighborhood.reshape(-1, channels)
            median_vector = np.median(vectors, axis=0)
            filtered_img[i, j] = median_vector
    return filtered_img


input_img = cv2.imread("buildings.jpg")
filtered_img = vector_median_filter(input_img, 3)
cv2.imshow("Filtered", filtered_img)
cv2.waitKey(0)

# This is quite slow (takes more than 2 minutes)