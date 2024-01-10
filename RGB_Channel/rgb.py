import cv2
import numpy as np

image = cv2.imread('D:/project/Cam/cycle/grayScale/avtr3.png')

if image is None:
    print("Error: Could not read the image.")
    exit()

# Create an empty list to store the separated channels
channels = []

# Split the image into its RGB channels
for i in range(3):
    channels.append(image[:, :, i])

for i, channel in enumerate(channels):
    # Create a window to display the channel
    cv2.namedWindow(f"Channel {i}")

    # Display the channel
    cv2.imshow(f'Channel {i}', channel)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

