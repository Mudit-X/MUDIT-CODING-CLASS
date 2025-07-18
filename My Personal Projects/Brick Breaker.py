import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Game constants
WIDTH, HEIGHT = 640, 480
PADDLE_WIDTH, PADDLE_HEIGHT = 100, 10
BALL_RADIUS = 10
BRICK_ROWS = 3
BRICK_COLS = 7
BRICK_WIDTH = WIDTH // BRICK_COLS
BRICK_HEIGHT = 20

# Paddle
paddle_x = WIDTH // 2 - PADDLE_WIDTH // 2
paddle_y = HEIGHT - 30

# Ball
ball_x, ball_y = WIDTH // 2, HEIGHT // 2
ball_dx, ball_dy = 4, -4

# Bricks
bricks = np.ones((BRICK_ROWS, BRICK_COLS), dtype=bool)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Detect hand and move paddle
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
            paddle_x = int(x * WIDTH - PADDLE_WIDTH / 2)
            paddle_x = np.clip(paddle_x, 0, WIDTH - PADDLE_WIDTH)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Update ball position
    ball_x += ball_dx
    ball_y += ball_dy

    # Wall collision
    if ball_x - BALL_RADIUS <= 0 or ball_x + BALL_RADIUS >= WIDTH:
        ball_dx *= -1
    if ball_y - BALL_RADIUS <= 0:
        ball_dy *= -1

    # Paddle collision
    if paddle_y <= ball_y + BALL_RADIUS <= paddle_y + PADDLE_HEIGHT and paddle_x <= ball_x <= paddle_x + PADDLE_WIDTH:
        ball_dy *= -1

    # Brick collision
    row = ball_y // BRICK_HEIGHT
    col = ball_x // BRICK_WIDTH
    if row < BRICK_ROWS and bricks[row][col]:
        bricks[row][col] = False
        ball_dy *= -1

    # Game over
    if ball_y > HEIGHT:
        cv2.putText(frame, "Game Over!", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        cv2.imshow("Brick Breaker", frame)
        if cv2.waitKey(2000) & 0xFF == ord('q'):
            break
        ball_x, ball_y = WIDTH // 2, HEIGHT // 2
        ball_dy *= -1
        bricks = np.ones((BRICK_ROWS, BRICK_COLS), dtype=bool)

    # Draw bricks
    for r in range(BRICK_ROWS):
        for c in range(BRICK_COLS):
            if bricks[r][c]:
                cv2.rectangle(frame, (c * BRICK_WIDTH, r * BRICK_HEIGHT),
                              ((c + 1) * BRICK_WIDTH, (r + 1) * BRICK_HEIGHT), (0, 255, 0), -1)

    # Draw paddle and ball
    cv2.rectangle(frame, (paddle_x, paddle_y), (paddle_x + PADDLE_WIDTH, paddle_y + PADDLE_HEIGHT), (255, 0, 0), -1)
    cv2.circle(frame, (ball_x, ball_y), BALL_RADIUS, (0, 0, 255), -1)

    cv2.imshow("Brick Breaker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
