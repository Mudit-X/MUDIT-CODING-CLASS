import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained emotion detection model
emotion_model_path = 'emotion_model.h5'
emotion_model = load_model(emotion_model_path)

# Load Haar cascades for face and (optionally) eye detection
face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
eye_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

if face_cascade.empty():
    print(f"Error: Face cascade XML file not found at {face_cascade_path}")
else:
    print("Face cascade loaded successfully!")
if eye_cascade.empty():
    print(f"Error: Eye cascade XML file not found at {eye_cascade_path}")
else:
    print("Eye cascade loaded successfully!")

# Emotion labels based on the model's output order
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting real-time face and emotion detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle

        face_roi = gray_frame[y:y + h, x:x + w]

        try:
            resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
            processed_face = np.expand_dims(np.expand_dims(resized_face, -1), 0) / 255.0  # Shape: (1, 48, 48, 1)
        except Exception as e:
            print(f"Warning: Could not process face ROI. Error: {e}")
            continue

        # Predict emotion
        emotion_prediction = emotion_model.predict(processed_face)
        predicted_emotion_index = np.argmax(emotion_prediction)
        predicted_emotion = emotion_labels[predicted_emotion_index]
        confidence = np.max(emotion_prediction)

        # Display label and confidence
        label = f"{predicted_emotion} ({confidence:.2f})"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the output frame
    cv2.imshow('Face and Emotion Detection', frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
