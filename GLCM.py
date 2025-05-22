import cv2 as cv
import numpy as np

# I am taking d=1 and theta = 0 degrees

def compute_glcm_0deg(image, levels):
    rows, cols = image.shape
    glcm = np.zeros((levels, levels), dtype=np.uint32)

    for i in range(rows):
        for j in range(cols - 1):
            row_val = image[i, j]
            col_val = image[i, j + 1]
            glcm[row_val, col_val] += 1

    return glcm

if __name__ == "__main__":
    img = cv.imread("buildings.jpg", cv.IMREAD_GRAYSCALE)
    glcm = compute_glcm_0deg(img, 256)
    print(glcm)
