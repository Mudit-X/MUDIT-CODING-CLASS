#Color Conversions and Cropping
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('L08/example2.jpg')

#Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.title("RGB Image")
plt.show()

#Convert to Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image, cmap='gray')
plt.title("Grayscale Image")
plt.show()

#Cropping the image
#Asuume we know the region we want: rows 200 to 300, columns 300 to 500
cropped_image = image[200:300, 300:500]
cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
plt.imshow(cropped_rgb)
plt.title("Cropped Region")
plt.show()