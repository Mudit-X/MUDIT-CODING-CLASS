import cv2
import numpy as np 
import os
from tensorflow.keras.models import load_model

# Load Haar cascade for face detection
face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    print(f"ERROR: Could not load face cascade XML file from {face_cascade_path}")
    exit()

# Path to your trained emotion recognition model
emotion_model_path = r"C:\Users\Admin\OneDrive\Desktop\MUDIT CODING CLASS\L12\emotion_model.h5"
print("Looking for model at:", os.path.abspath(emotion_model_path))

# Load the emotion recognition model with compile=False to avoid old 'lr' error
try:
    emotion_model = load_model(emotion_model_path, compile=False)
except Exception as e:
    print(f"ERROR: Could not load emotion recognition model from {emotion_model_path}")
    print(f"Details: {e}")
    exit()

# List of emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start webcam
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
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Blue rectangle
        face_roi = gray_frame[y:y+h, x:x+w]

        try:
            resized_face = cv2.resize(face_roi, (64, 64), interpolation=cv2.INTER_AREA)
            processed_face = np.expand_dims(np.expand_dims(resized_face, -1), 0) / 255.0
        except Exception as e:
            print(f"Warning: Could not process face ROI. Error: {e}")
            continue

        # Predict emotion
        emotion_prediction = emotion_model.predict(processed_face)
        predicted_emotion_index = np.argmax(emotion_prediction)
        predicted_emotion = emotion_labels[predicted_emotion_index]

        # Display label
        cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Face and Emotion Detection', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()