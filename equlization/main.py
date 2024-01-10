import cv2
import numpy as np

image = cv2.imread('D:/project/Cam/cycle/grayScale/avtr3_grayscale_image.jpg', cv2.IMREAD_GRAYSCALE)

equalized_image = cv2.equalizeHist(image)

# Display the original grayscale image
cv2.imshow('Original Grayscale Image', image)
cv2.waitKey(0)

# Display the equalized grayscale image
cv2.imshow('Equalized Grayscale Image', equalized_image)
cv2.waitKey(0)

# Close all the windows
cv2.destroyAllWindows()