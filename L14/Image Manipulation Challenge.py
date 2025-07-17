import cv2
import numpy as np
import os

# ---------- STEP 1: Load the Image ----------
image_path = r'C:\Users\Admin\OneDrive\Desktop\MUDIT CODING CLASS\L14\IMG_5491.jpg'  # Use raw string (r'...') to avoid unicode errors
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found.")
    exit()

# ---------- STEP 2: Rotate the Image ----------
def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rotation_matrix, (w, h))

rotated = rotate_image(image, angle=0)  # Change angle as needed

# ---------- STEP 3: Adjust Brightness ----------
def adjust_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v + value, 0, 255).astype(np.uint8)
    bright_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(bright_hsv, cv2.COLOR_HSV2BGR)

brightened = adjust_brightness(rotated, value=40)

# ---------- STEP 4: Crop the Main Subject ----------
def crop_center(img, percent=0.5):
    h, w = img.shape[:2]
    new_h, new_w = int(h * percent), int(w * percent)
    start_y, start_x = (h - new_h) // 2, (w - new_w) // 2
    return img[start_y:start_y+new_h, start_x:start_x+new_w]

cropped = crop_center(brightened, percent=0.5)

# ---------- STEP 5: Resize for Display ----------
def resize_for_display(img, max_width=800):
    h, w = img.shape[:2]
    if w > max_width:
        scaling_factor = max_width / w
        new_size = (int(w * scaling_factor), int(h * scaling_factor))
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img

# ---------- STEP 6: Save the Final Image ----------
output_path = os.path.join(os.getcwd(), 'edited_image.jpg')
cv2.imwrite(output_path, cropped)
print(f"Image successfully saved as '{output_path}'.")

# ---------- STEP 7: Display Each Step (Resized) ----------
cv2.imshow("Original", resize_for_display(image))
cv2.imshow("Rotated", resize_for_display(rotated))
cv2.imshow("Brightened", resize_for_display(brightened))
cv2.imshow("Cropped", resize_for_display(cropped))
cv2.waitKey(0)
cv2.destroyAllWindows()
