import cv2

# Read the image in grayscale
image = cv2.imread('D:/project/Cam/cycle/grayScale/avtr3_grayscale_image.jpg', cv2.IMREAD_GRAYSCALE)

# Detect edges using the Canny algorithm
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(image, low_threshold, high_threshold)

# Display the original grayscale image
cv2.imshow('Original Grayscale Image', image)
cv2.waitKey(0)

# Display the edge map
cv2.imshow('Edge Map', edges)
cv2.waitKey(0)

# Close all the windows
cv2.destroyAllWindows()

