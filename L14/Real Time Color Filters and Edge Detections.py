import cv2
import numpy as np

def apply_filter(image, filter_type):
    """Apply the specified color filter to the image"""
    #Create a copy of the image to avoid modifying the original
    filtered_image = image.copy()

    if filter_type == "red_tint":
        filtered_image[:, :, 1] = 0  #Green channel to 0
        filtered_image[:, :, 0] = 0  #Blue channel to 0

    elif filter_type == "blue_tint":
        filtered_image[:, :, 1] = 0  #Green channel to 0
        filtered_image[:, :, 2] = 0  #Red channel to 0

    elif filter_type == "green_tint":
        filtered_image[:, :, 0] = 0  #Blue channel to 0
        filtered_image[:, :, 2] = 0  #Red channel to 0

    elif filter_type == "sobel":
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
        combined_sobel = cv2.bitwise_or(sobel_x.astype('uint8'), sobel_y.astype('uint8'))
        filtered_image = cv2.cvtColor(combined_sobel, cv2.COLOR_GRAY2BGR)

    elif filter_type == "canny":
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 100, 200)
        filtered_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return filtered_image

#Load the image
image_path = 'L14/example.jpg'  #Provide your image path
image = cv2.imread(image_path)

if image is None:
    print("Error: image not found")
else:
    filter_type = "original"

    print("Press the following keys to apply filters:")
    print("r - Red Tint")
    print("b - Blue Tint")
    print("g - Green Tint")
    print("s - Sobel Edge Detection")
    print("c - Canny Edge Detection")
    print("q - Quit")

    while True:
        #Apply the selected filter
        filtered_image = apply_filter(image, filter_type)

        #Display the filtered image
        cv2.imshow("Filtered Image", filtered_image)

        #Wait for key press
        key = cv2.waitKey(0) & 0xFF

        #Map for key press
        if key == ord('r'):
            filter_type = "red_tint"
        elif key == ord('b'):
            filter_type = "blue_tint"
        elif key == ord('g'):
            filter_type = "green_tint"
        elif key == ord('s'):
            filter_type = "sobel"
        elif key == ord('c'):
            filter_type = "canny"
        elif key == ord('q'):
            print("Exiting...")
            break
        else:
            print("Invalid key! Use 'r', 'b', 'g', 's', 'c', or 'q'.")

cv2.destroyAllWindows()