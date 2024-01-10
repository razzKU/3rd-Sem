import cv2
import numpy as np

def apply_morphological_operations(image):
    kernel = np.ones((5, 5), np.uint8)

    dilation = cv2.dilate(image, kernel, iterations=1)
    erosion = cv2.erode(image, kernel, iterations=1)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return dilation, erosion, opening, closing

image = cv2.imread('D:/project/Cam/cycle/grayScale/avtr3.png', 0)
dilation, erosion, opening, closing = apply_morphological_operations(image)

cv2.imshow('Original Image', image)
cv2.imshow('Dilation', dilation)
cv2.imshow('Erosion', erosion)
cv2.imshow('Opening', opening)
cv2.imshow('Closing', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()