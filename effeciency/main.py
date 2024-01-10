import cv2
import numpy as np

def threshold_segmentation(image, threshold=127):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return thresholded

image = cv2.imread('D:/project/Cam/cycle/grayScale/avtr3_grayscale_image.jpg')
segmented_image = threshold_segmentation(image)

cv2.imshow('Original Image', image)
cv2.imshow('Threshold Segmentation', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def otsu_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

image = cv2.imread('D:/project/Cam/cycle/grayScale/avtr3_grayscale_image.jpg')
segmented_image = otsu_segmentation(image)

cv2.imshow('Original Image', image)
cv2.imshow('Otsu Segmentation', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()