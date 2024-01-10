import cv2
import matplotlib.pyplot as plt

gray_image =  cv2.imread('D:/project/Cam/cycle/grayScale/avtr3_grayscale_image.jpg')

# Calculate the histogram of the grayscale image
histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# Create a plot with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2)

# Display the original grayscale image on the first subplot
ax1.imshow(gray_image, cmap='gray')
ax1.set_title('Original Grayscale Image')

# Display the histogram on the second subplot
ax2.plot(histogram)
ax2.set_title('Histogram of Grayscale Image')

# Display the plot
plt.show()

