import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Game window size
screen_width = 960
screen_height = 540

# Ball settings
ball_radius = 20
initial_speed = 5
balls = [{'x': random.randint(0, screen_width), 'y': 0, 'speed': initial_speed}]
ball_color = (0, 0, 255)

# Paddle settings
paddle_width = 100
paddle_height = 20
score = 0

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, screen_width)
cap.set(4, screen_height)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Default paddle position
    paddle_x = screen_width // 2
    paddle_y = screen_height - 50

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

            # Index finger tip landmark
            index = hand_landmark.landmark[8]
            paddle_x = int(index.x * screen_width)

    # Draw paddle (rectangle)
    top_left = (paddle_x - paddle_width // 2, paddle_y - paddle_height // 2)
    bottom_right = (paddle_x + paddle_width // 2, paddle_y + paddle_height // 2)
    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), -1)

    # Move and draw balls
    for ball in balls:
        ball['y'] += ball['speed']
        cv2.circle(frame, (ball['x'], ball['y']), ball_radius, ball_color, -1)

        # Check collision
        if (top_left[0] <= ball['x'] <= bottom_right[0]) and (top_left[1] <= ball['y'] + ball_radius <= bottom_right[1]):
            score += 1
            ball['x'] = random.randint(0, screen_width)
            ball['y'] = 0

            # Add new ball every 10 catches
            if score % 10 == 0:
                balls.append({'x': random.randint(0, screen_width), 'y': 0, 'speed': initial_speed})

            # Increase speed every 5 catches
            if score % 5 == 0:
                for b in balls:
                    b['speed'] += 1

        # Reset ball if it falls off screen
        if ball['y'] > screen_height:
            ball['x'] = random.randint(0, screen_width)
            ball['y'] = 0

    # Display score
    cv2.putText(frame, f"Score: {score}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Catch the Falling Balls 🎮", frame)

    # Quit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
