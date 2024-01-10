import cv2
import numpy as np

image = cv2.imread('D:/project/Cam/cycle/grayScale/avtr3.png')

if image is None:
    print("Error: Could not read the image.")
    exit()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a threshold to the grayscale image
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Create a window to display the original grayscale image
cv2.namedWindow('Original Grayscale Image')

# Display the original grayscale image
cv2.imshow('Original Grayscale Image', gray_image)

# Create a window to display the binary image
cv2.namedWindow('Binary Image')

# Display the binary image
cv2.imshow('Binary Image', binary_image)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()