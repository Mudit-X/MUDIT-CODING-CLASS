import cv2
import matplotlib.pyplot as plt

#Step 1: Load the image
image_path = 'L09/example_2.jpg'
image = cv2.imread(image_path)

# --- IMPORTANT: Always check if the image loaded successfully ---
if image is None:
    print(f"Error: Could not load image from '{image_path}'.")
    print("Please ensure the image file exists at the specified path.")
    exit()

# Convert BGR to RGB for correct color display with matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get image dimensions
height, width, _ = image_rgb.shape

# --- MODIFICATIONS START HERE ---

# Step 2: Draw Two Rectangles Around Interesting Regions
# Make shapes BIGGER and change colors/thickness
rect1_width, rect1_height = 250, 250  # Increased size from 150, 150
top_left1 = (20, 20)
bottom_right1 = (top_left1[0] + rect1_width, top_left1[1] + rect1_height)
# Changed color to Bright Red (255, 0, 0) and thickness to 5 (from 3)
cv2.rectangle(image_rgb, top_left1, bottom_right1, (255, 0, 0), 5)

# Rectangle 2: Bottom-right corner
rect2_width, rect2_height = 300, 250  # Increased size from 200, 150
top_left2 = (width - rect2_width - 20, height - rect2_height - 20)
bottom_right2 = (top_left2[0] + rect2_width, top_left2[1] + rect2_height)
# Changed color to Deep Blue (0, 0, 200) and thickness to 5 (from 3)
cv2.rectangle(image_rgb, top_left2, bottom_right2, (0, 0, 200), 5)

# Step 3: Draw Circles at the Centers of Both Rectangles
center1_x = top_left1[0] + rect1_width // 2
center1_y = top_left1[1] + rect1_height // 2

center2_x = top_left2[0] + rect2_width // 2
center2_y = top_left2[1] + rect2_height // 2

# Increased radius to 30 (from 15) and changed color to Orange (255, 165, 0)
cv2.circle(image_rgb, (center1_x, center1_y), 30, (255, 165, 0), -1) # Filled Orange Circle
# Increased radius to 30 (from 15) and changed color to Purple (128, 0, 128)
cv2.circle(image_rgb, (center2_x, center2_y), 30, (128, 0, 128), -1) # Filled Purple Circle

# Step 4: Draw Connecting Lines Between Centers of Rectangles
# Changed color to Cyan (0, 255, 255) and thickness to 4 (from 2)
cv2.line(image_rgb, (center1_x, center1_y), (center2_x, center2_y), (0, 255, 255), 4, cv2.LINE_AA)

# Step 5: Add Text Labels for Regions and Centers
font = cv2.FONT_HERSHEY_SIMPLEX # Keeping the font type

# Region 1 Text: Increased font scale to 1.0 (from 0.7), color to Yellow (255, 255, 0), thickness to 3 (from 2)
cv2.putText(image_rgb, 'Region 1', (top_left1[0], top_left1[1] - 15), font, 1.0, (255, 255, 0), 3, cv2.LINE_AA)
# Region 2 Text: Increased font scale to 1.0 (from 0.7), color to Pink (255, 105, 180), thickness to 3 (from 2)
cv2.putText(image_rgb, 'Region 2', (top_left2[0], top_left2[1] - 15), font, 1.0, (255, 105, 180), 3, cv2.LINE_AA)

# Center 1 Text: Increased font scale to 0.9 (from 0.6), color to White (255, 255, 255), thickness to 2 (from 2)
cv2.putText(image_rgb, 'Center 1', (center1_x - 60, center1_y + 50), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
# Center 2 Text: Increased font scale to 0.9 (from 0.6), color to Light Blue (173, 216, 230), thickness to 2 (from 2)
cv2.putText(image_rgb, 'Center 2', (center2_x - 60, center2_y + 50), font, 0.9, (173, 216, 230), 2, cv2.LINE_AA)

# Step 6: Add Bi-Directional Arrow Representing Height
arrow_start = (width - 50, 20) # Start near the top-right
arrow_end = (width - 50, height - 20) # End near the bottom-right

# Draw arrows in both directions
# Changed color to Bright Green (0, 255, 0) and thickness to 6 (from 3)
cv2.arrowedLine(image_rgb, arrow_start, arrow_end, (0, 255, 0), 6, tipLength=0.05) # Downward arrow
cv2.arrowedLine(image_rgb, arrow_end, arrow_start, (0, 255, 0), 6, tipLength=0.05) # Upward arrow

# Annotate the height value
height_label_position = (arrow_start[0] - 180, (arrow_start[1] + arrow_end[1]) // 2) # Adjusted position for larger text
# Height Text: Increased font scale to 1.0 (from 0.8), color to Green Yellow (173, 255, 47), thickness to 3 (from 2)
cv2.putText(image_rgb, f'Height: {height}px', height_label_position, font, 1.0, (173, 255, 47), 3, cv2.LINE_AA)

# --- MODIFICATIONS END HERE ---

# Step 7: Display the Annotated Image
plt.figure(figsize=(14, 10)) # Increased figure size for better viewing
plt.imshow(image_rgb)
plt.title('Annotated Image with Enhanced Regions, Centers, and Bi-Directional Height Arrow')
plt.axis('off')
plt.show()