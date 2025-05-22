import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# I am using K-means clustering for this

def kmeans_segmentation(image_path, k):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
cv2.imshow('segmented image',segmented_img)
cv2.waitKey(0)
    