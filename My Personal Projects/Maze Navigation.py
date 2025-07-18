import cv2
import mediapipe as mp
import numpy as np
import random

# Constants
ROWS, COLS = 9, 16
TILE_SIZE = 60
WIDTH, HEIGHT = COLS * TILE_SIZE, ROWS * TILE_SIZE

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Maze Generation (random walls)
def generate_maze():
    maze = np.zeros((ROWS, COLS), dtype=np.uint8)
    for i in range(ROWS):
        for j in range(COLS):
            if random.random() < 0.2 and (i, j) != (0, 0):
                maze[i][j] = 1
    return maze

# Gesture classifier
def classify_gesture(landmarks):
    fingers = []

    # Thumb
    if landmarks[4].x < landmarks[3].x:
        fingers.append('thumb')  # RIGHT

    # Index
    if landmarks[8].y < landmarks[6].y:
        fingers.append('index')  # DOWN

    # Pinky
    if landmarks[20].x > landmarks[19].x:
        fingers.append('pinky')  # LEFT

    # All open = UP
    if (landmarks[4].x < landmarks[3].x and
        landmarks[8].y < landmarks[6].y and
        landmarks[12].y < landmarks[10].y and
        landmarks[16].y < landmarks[14].y and
        landmarks[20].y < landmarks[18].y):
        return 'up'

    if 'thumb' in fingers: return 'right'
    if 'index' in fingers: return 'down'
    if 'pinky' in fingers: return 'left'
    return 'none'

# Initialize game state
maze = generate_maze()
pos = [0, 0]
last_move = None

# Capture from webcam
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    gesture = 'none'

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            gesture = classify_gesture(handLms.landmark)

    # Movement logic: One space at a time
    if gesture != last_move:
        r, c = pos
        if gesture == 'right' and c + 1 < COLS and maze[r][c + 1] == 0:
            pos[1] += 1
        elif gesture == 'left' and c - 1 >= 0 and maze[r][c - 1] == 0:
            pos[1] -= 1
        elif gesture == 'up' and r - 1 >= 0 and maze[r - 1][c] == 0:
            pos[0] -= 1
        elif gesture == 'down' and r + 1 < ROWS and maze[r + 1][c] == 0:
            pos[0] += 1
        last_move = gesture

    # Draw Maze
    display = np.ones((HEIGHT, WIDTH, 3), np.uint8) * 255
    for i in range(ROWS):
        for j in range(COLS):
            color = (0, 0, 0) if maze[i][j] == 1 else (200, 200, 200)
            cv2.rectangle(display, (j*TILE_SIZE, i*TILE_SIZE),
                          ((j+1)*TILE_SIZE, (i+1)*TILE_SIZE), color, -1)

    # Draw Player
    cv2.circle(display, (pos[1]*TILE_SIZE + TILE_SIZE//2, pos[0]*TILE_SIZE + TILE_SIZE//2), TILE_SIZE//3, (0, 0, 255), -1)

    # Combine camera feed + game display
    cam_resized = cv2.resize(img, (WIDTH, HEIGHT))
    combined = np.hstack((cam_resized, display))

    cv2.imshow("Maze Game - Left: Camera | Right: Maze", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
