import cv2
import time
import pyautogui
import mediapipe as mp

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Configuration
SCROLL_SPEED = 300
SCROLL_DELAY = 1
CAM_WIDTH, CAM_HEIGHT = 640, 480

def detect_gesture(landmarks, handedness):
    fingers = []

    # Tip indices for 4 fingers (excluding thumb)
    tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]

    # Check if fingers are open (relaxed threshold)
    for tip in tips:
        pip = tip - 2
        if landmarks.landmark[tip].y < landmarks.landmark[pip].y - 0.01:  # more forgiving
            fingers.append(1)
        else:
            fingers.append(0)

    # Thumb logic (same as before)
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

    if (handedness == "Right" and thumb_tip.x > thumb_mcp.x) or \
       (handedness == "Left" and thumb_tip.x < thumb_mcp.x):
        fingers.append(1)
    else:
        fingers.append(0)

    total_fingers = sum(fingers)

    # Gesture rules (relaxed palm)
    if total_fingers >= 4:
        return "scroll_up", fingers
    elif total_fingers == 0:
        return "scroll_down", fingers
    else:
        return "none", fingers

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)
last_scroll = 0
p_time = time.time()

print("Gesture Scroll Control Active\nOpen palm (relaxed): Scroll Up\nFist: Scroll Down\nPress 'q' to exit")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    gesture, handedness, fingers = "none", "Unknown", []

    if results.multi_hand_landmarks:
        for hand, handedness_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            handedness = handedness_info.classification[0].label
            gesture, fingers = detect_gesture(hand, handedness)
            mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

    # Scroll action
    if (time.time() - last_scroll) > SCROLL_DELAY:
        if gesture == "scroll_up":
            pyautogui.scroll(SCROLL_SPEED)
            last_scroll = time.time()
        elif gesture == "scroll_down":
            pyautogui.scroll(-SCROLL_SPEED)
            last_scroll = time.time()

    # FPS
    fps = 1 / (time.time() - p_time) if (time.time() - p_time) > 0 else 0
    p_time = time.time()

    # Flip image for mirror effect
    flipped_img = cv2.flip(img, 1)

    # Display info
    cv2.putText(flipped_img, f"FPS: {int(fps)} | Hand: {handedness} | Gesture: {gesture}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(flipped_img, f"Fingers: {fingers}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Gesture Control", flipped_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
