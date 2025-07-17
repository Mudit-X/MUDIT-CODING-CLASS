import cv2
import numpy as np
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import math
import screen_brightness_control as sbc

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Setup webcam
cap = cv2.VideoCapture(0)

# Setup Pycaw (volume control)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

def fingers_up(lm_list):
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]

    # Thumb
    if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for id in range(1, 5):
        if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, c = img.shape
    exit_detected = False

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            lm_list = []
            hand_label = hand_handedness.classification[0].label  # 'Left' or 'Right'

            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            if len(lm_list) >= 21:
                fingers = fingers_up(lm_list)

                # Detect "Yow" (shaka 🤙): only thumb and pinky up
                if fingers == [1, 0, 0, 0, 1]:
                    exit_detected = True

                x1, y1 = lm_list[4][1], lm_list[4][2]
                x2, y2 = lm_list[8][1], lm_list[8][2]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

                length = math.hypot(x2 - x1, y2 - y1)

                if hand_label == 'Right':
                    vol_scalar = np.interp(length, [30, 250], [0.0, 1.0])
                    volume.SetMasterVolumeLevelScalar(vol_scalar, None)
                    vol_bar = int(np.interp(length, [30, 250], [400, 150]))
                    cv2.putText(img, 'Volume', (50, 430),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 2)
                    cv2.rectangle(img, (50, vol_bar), (85, 400), (0, 255, 0), cv2.FILLED)

                elif hand_label == 'Left':
                    brightness = int(np.interp(length, [30, 250], [0, 100]))
                    try:
                        sbc.set_brightness(brightness)
                    except Exception as e:
                        print("Brightness control error:", e)
                    bright_bar = int(np.interp(length, [30, 250], [400, 150]))
                    cv2.putText(img, 'Brightness', (550, 430),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    cv2.rectangle(img, (550, 150), (585, 400), (255, 255, 0), 2)
                    cv2.rectangle(img, (550, bright_bar), (585, 400), (255, 255, 0), cv2.FILLED)

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q') or exit_detected:
        break

cap.release()
cv2.destroyAllWindows()
