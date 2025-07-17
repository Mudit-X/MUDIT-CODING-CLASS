import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize webcam
cap = cv2.VideoCapture(0)

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Game variables
score = 0
target_radius = 40
target_position = (random.randint(100, 540), random.randint(100, 380))
hit_cooldown = 0

# Game state
game_over = False

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Draw UI
    if not game_over:
        cv2.circle(frame, target_position, target_radius, (0, 0, 255), -1)
        cv2.putText(frame, f"Score: {score}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    else:
        cv2.putText(frame, "GAME OVER", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.putText(frame, "Thumbs up to restart", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if hit_cooldown > 0:
        hit_cooldown -= 1

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            landmarks = handLms.landmark

            # Get finger coordinates
            index_tip = landmarks[8]
            thumb_tip = landmarks[4]

            index_x = int(index_tip.x * w)
            index_y = int(index_tip.y * h)
            thumb_x = int(thumb_tip.x * w)
            thumb_y = int(thumb_tip.y * h)

            # Draw pointer
            cv2.circle(frame, (index_x, index_y), 10, (255, 255, 0), -1)

            if not game_over:
                # Check hit
                distance = np.linalg.norm(np.array([index_x, index_y]) - np.array(target_position))
                if distance < target_radius and hit_cooldown == 0:
                    score += 1
                    hit_cooldown = 10
                    target_position = (random.randint(100, 540), random.randint(100, 380))

                # End the game if score hits 5
                if score >= 11:
                    game_over = True
            else:
                # Restart on thumbs up
                if thumb_tip.y < landmarks[3].y and all(landmarks[i].y > landmarks[i-2].y for i in [6, 10, 14, 18]):
                    score = 0
                    target_position = (random.randint(100, 540), random.randint(100, 380))
                    game_over = False

    cv2.imshow("Gesture Game", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
