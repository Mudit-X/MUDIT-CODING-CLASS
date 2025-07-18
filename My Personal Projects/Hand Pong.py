import cv2
import mediapipe as mp
import numpy as np

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Game settings
width, height = 1280, 720
paddle_height = 150
paddle_width = 20
ball_radius = 20
paddle_speed = 10
ball_speed_x, ball_speed_y = 10, 10

# Initial positions
left_paddle_y = right_paddle_y = height // 2
ball_x, ball_y = width // 2, height // 2
score_left, score_right = 0, 0

# Capture
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

def detect_hands(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    positions = []
    if results.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hand_info.classification[0].label  # 'Left' or 'Right'
            cx = int(hand_landmarks.landmark[0].x * width)
            cy = int(hand_landmarks.landmark[0].y * height)
            positions.append((label, cx, cy))
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return positions

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (width, height))

    hand_positions = detect_hands(frame)

    for label, cx, cy in hand_positions:
        if label == "Left" and cx < width // 2:
            left_paddle_y = cy
        elif label == "Right" and cx > width // 2:
            right_paddle_y = cy

    # Update ball position
    ball_x += ball_speed_x
    ball_y += ball_speed_y

    # Ball bounce top/bottom
    if ball_y - ball_radius < 0 or ball_y + ball_radius > height:
        ball_speed_y *= -1

    # Paddle collision
    if (ball_x - ball_radius < 40 and abs(ball_y - left_paddle_y) < paddle_height // 2) or \
       (ball_x + ball_radius > width - 40 and abs(ball_y - right_paddle_y) < paddle_height // 2):
        ball_speed_x *= -1

    # Scoring
    if ball_x < 0:
        score_right += 1
        ball_x, ball_y = width // 2, height // 2
        ball_speed_x = -ball_speed_x

    elif ball_x > width:
        score_left += 1
        ball_x, ball_y = width // 2, height // 2
        ball_speed_x = -ball_speed_x

    # Draw paddles
    cv2.rectangle(frame, (20, left_paddle_y - paddle_height // 2), (20 + paddle_width, left_paddle_y + paddle_height // 2), (255, 0, 0), -1)
    cv2.rectangle(frame, (width - 40, right_paddle_y - paddle_height // 2), (width - 40 + paddle_width, right_paddle_y + paddle_height // 2), (0, 255, 0), -1)

    # Draw ball
    cv2.circle(frame, (ball_x, ball_y), ball_radius, (0, 0, 255), -1)

    # Draw score
    cv2.putText(frame, f"{score_left} : {score_right}", (width // 2 - 70, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

    cv2.imshow("Hand Pong", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()