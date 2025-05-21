import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def rescale_image(image, scale):
    width = int(image.shape[1]*scale)
    height = int(image.shape[0]*scale)
    dimension =(width, height)
    return cv.resize(image, dimension, interpolation=cv.INTER_AREA)

def im2col(img, block_size):
    image_block = []
    block_width, block_height = block_size  
    
    for i in range(0, img.shape[0], block_width):
        for j in range(0, img.shape[1], block_height):
            block = img[i:i+block_width, j:j+block_height]
            if block.shape == (block_width, block_height): 
                image_block.append(block.reshape(-1))
    
    return np.array(image_block)

def col2im(mtx, image_size, block_size):
    block_width, block_height = block_size
    sx, sy = image_size[1], image_size[0]
    result = np.zeros(image_size)
    col = 0
    
    for i in range(0, sy, block_width):
        for j in range(0, sx, block_height):
            if col < mtx.shape[1]: 
                block = mtx[:, col].reshape(block_width, block_height)
                result[i:i+block_width, j:j+block_height] = block
                col += 1
    return result

def kl_transform_blocks(blocks, k=10):
    mean = np.mean(blocks, axis=0)  
    centered_blocks = blocks - mean
    covariance = np.cov(centered_blocks, rowvar=False)  
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    compressed_blocks = np.matmul(centered_blocks, eigenvectors[:, :k])
    return compressed_blocks, eigenvectors[:, :k], mean

def reconstruct_blocks(compressed_blocks, eigenvectors, mean):
    return np.matmul(compressed_blocks, eigenvectors.T) + mean

img = cv.imread('buildings.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = rescale_image(img ,0.25)
img = img.astype(np.double)
img = img/255
# cv.imshow('x',img)
# cv.waitKey(0)
blocks = im2col(img, (8, 8))  
compressed, evecs, mean = kl_transform_blocks(blocks, 10) # using only top 10 evectors
reconstructed_blocks = reconstruct_blocks(compressed, evecs, mean)
# reconstructed_img = col2im(reconstructed_blocks, img.shape, (8, 8)) -> something going wrong here

# cv.imshow("x",reconstructed_img)
# cv.waitKey(0)