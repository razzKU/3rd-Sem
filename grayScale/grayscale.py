import cv2
import numpy as np

image = cv2.imread('D:/project/Cam/cycle/grayScale/avtr3.png')

if image is None:
    print("Error: Could not read the image.")
    exit()

def rgb_to_grayscale(image):
    """
    Converts an RGB image into grayscale.
    """
    # Get the dimensions of the image
    height, width, channels = image.shape

    # Create a new grayscale image
    grayscale_image = np.zeros((height, width), dtype=np.uint8)

    # Calculate the grayscale pixel values
    for y in range(height):
        for x in range(width):
            # Get the RGB pixel values
            r, g, b = image[y, x]

            # Calculate the grayscale pixel value
            gray = (r * 0.299) + (g * 0.587) + (b * 0.114)

            # Assign the grayscale pixel value to the grayscale image
            grayscale_image[y, x] = gray

    return grayscale_image

grayscale_image = rgb_to_grayscale(image)

cv2.imwrite('D:/project/Cam/cycle/grayScale/avtr3_grayscale_image.jpg', grayscale_image)