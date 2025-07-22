import cv2
import mediapipe as mp

# Initialize mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Drawing specs
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw only eyes landmarks (around index 33–133)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

    # Show frame
    cv2.imshow('Eye Tracking - MediaPipe', image)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
