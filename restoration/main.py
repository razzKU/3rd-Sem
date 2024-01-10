import numpy as np
import cv2
import matplotlib.pyplot as plt

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    total_pixels = image.size

    # Add salt noise
    num_salt = np.ceil(salt_prob * total_pixels)
    salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 1

    # Add pepper noise
    num_pepper = np.ceil(pepper_prob * total_pixels)
    pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image

def mean_filter(image, filter_size):
    kernel = np.ones((filter_size, filter_size), np.float32) / (filter_size**2)
    result = cv2.filter2D(image, -1, kernel)
    return result

# Load an image
original_image = cv2.imread("D:/project/Cam/cycle/grayScale/avtr3.png", cv2.IMREAD_GRAYSCALE)

# Add salt and pepper noise
noisy_image = add_salt_and_pepper_noise(original_image, salt_prob=0.02, pepper_prob=0.02)

# Apply mean filter for image restoration
restored_image = mean_filter(noisy_image, filter_size=3)

# Display the images
plt.subplot(131), plt.imshow(original_image, cmap="gray"), plt.title("Original Image")
plt.subplot(132), plt.imshow(noisy_image, cmap="gray"), plt.title("Noisy Image")
plt.subplot(133), plt.imshow(restored_image, cmap="gray"), plt.title("Restored Image")
plt.show()
