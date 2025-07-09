import cv2
import numpy as np
import os

def apply_color_filter(image, filter_type):
    """Apply the specified color filter to the image"""
    filtered_image = image.copy()
    if filter_type == "red_tint":
        filtered_image[:, :, 1] = 0  # Green channel to 0
        filtered_image[:, :, 0] = 0  # Blue channel to 0
    elif filter_type == "blue_tint":
        filtered_image[:, :, 1] = 0  # Green channel to 0
        filtered_image[:, :, 2] = 0  # Red channel to 0
    elif filter_type == "green_tint":
        filtered_image[:, :, 0] = 0  # Blue channel to 0
        filtered_image[:, :, 2] = 0  # Red channel to 0
    elif filter_type == "increase_red":
        filtered_image[:, :, 2] = cv2.add(filtered_image[:, :, 2], 50)  # Increase red
    elif filter_type == "decrease_blue":
        filtered_image[:, :, 0] = cv2.subtract(filtered_image[:, :, 0], 50)  # Decrease blue
    return filtered_image

# Load the image
image_path = 'L11/example.jpg'  # Change path as needed
image = cv2.imread(image_path)

if image is None:
    print("Error: image not found")
else:
    filter_type = "original"
    print("Press the following keys to apply filters:")
    print("r - Red Tint")
    print("b - Blue Tint")
    print("g - Green Tint")
    print("i - Increase Red")
    print("d - Decrease Blue")
    print("s - Save the current Filtered Image")
    print("q - Quit")

    while True:
        filtered_image = apply_color_filter(image, filter_type)

        # Create resizable window and set size
        cv2.namedWindow("Filtered Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Filtered Image", 900, 600)

        # Show the image
        cv2.imshow("Filtered Image", filtered_image)

        # Wait for key press
        key = cv2.waitKey(0) & 0xFF

        # Handle key presses
        if key == ord('r'):
            filter_type = "red_tint"
        elif key == ord('b'):
            filter_type = "blue_tint"
        elif key == ord('g'):
            filter_type = "green_tint"
        elif key == ord('i'):
            filter_type = "increase_red"
        elif key == ord('d'):
            filter_type = "decrease_blue"
        elif key == ord('s'):
            # Save image to Downloads folder
            downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
            filename = f"filtered_{filter_type}.jpg"
            full_path = os.path.join(downloads_path, filename)
            cv2.imwrite(full_path, filtered_image)
            print(f"Image saved to: {full_path}")
        elif key == ord('q'):
            print("Exiting...")
            break
        else:
            print("Invalid key! Use 'r', 'b', 'g', 'i', 'd', 's', or 'q'.")

    cv2.destroyAllWindows()
