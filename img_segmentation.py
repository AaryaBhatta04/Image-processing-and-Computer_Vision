import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans
# I am using K-means clustering for this

def kmeans_segmentation(image_path, k):
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    height, width = image.shape[0], image.shape[1]
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pixel_values)
    centers = kmeans.cluster_centers_
    segmented_image = centers[labels].reshape((height, width, 3))
    segmented_image = np.uint8(segmented_image)
    mask = labels.reshape((height, width))
    
    return segmented_image, mask

    
segmented_img, cluster_mask = kmeans_segmentation('buildings.jpg', 4)
cv.imshow('segmented image',segmented_img)
cv.waitKey(0)
    