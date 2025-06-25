# Rotating and adjusting the brightness
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = 'L08/image_1.png'
image = cv2.imread(image_path)


# Convert BGR to RGB for consistent display with matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.title("Original RGB Image")
plt.show()

# Convert to Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image, cmap='gray')
plt.title("Grayscale Image")
plt.show()

# Rotate the image by 90 degrees around its center
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 90, 1.0) # Rotate by 90 degrees, scale 1.0
rotated = cv2.warpAffine(image, M, (w, h))

rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
plt.imshow(rotated_rgb)
plt.title("Rotated Image (90 degrees)")
plt.show()

# Increase Brightness by adding 60 to all pixel values
# Use cv2.add to avoid negative values or overflow (clips values to 0-255)
brightness_matrix = np.ones(image.shape, dtype="uint8") * 60
brighter = cv2.add(image, brightness_matrix)

bright_rgb = cv2.cvtColor(brighter, cv2.COLOR_BGR2RGB)
plt.imshow(bright_rgb)
plt.title("Brighter Image (+60 brightness)")
plt.show()

# Cropping the image
# Assumed region: rows 200 to 300, columns 200 to 300
cropped_image = image[200:300, 200:300] # Corrected: End values must be greater than start values
cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
plt.imshow(cropped_rgb)
plt.title("Cropped Region (Rows 200-300, Cols 200-300)") # Updated title for clarity
plt.show()