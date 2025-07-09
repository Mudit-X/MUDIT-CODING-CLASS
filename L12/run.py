import cv2

import os

# This will give you the path to the cv2 module

cv2_base_path = os.path.dirname(os.path.abspath(cv2.__file__))

# The cascades are typically in a 'data' subfolder

cascade_path = os.path.join(cv2_base_path, 'data', 'haarcascade_frontalface_default.xml')

print(f"Expected cascade path: {cascade_path}")

if os.path.exists(cascade_path):

 print("File found at this path!")

else:

 print("File NOT found at this path. You might need to manually locate or download it.")