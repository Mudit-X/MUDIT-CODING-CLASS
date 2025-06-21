import cv2

# Load the image
image = cv2.imread("L07/example_2.jpg")

#Window 1
cv2.namedWindow('Window 1', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Window 1', 900, 800)
cv2.imshow('Window 1', image)

#Window 2
cv2.namedWindow('Window 2', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Window 2', 700, 600)
cv2.imshow('Window 2', image)

#Window 3
cv2.namedWindow('Window 3', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Window 3', 500, 400)
cv2.imshow('Window 3', image)


cv2.waitKey(0)  # Wait for any key press
cv2.destroyAllWindows()  # Close all OpenCV windows

# Print image properties
print(f"Image Dimensions: {image.shape}")  # (Height, Width, Channels)