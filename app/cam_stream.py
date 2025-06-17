import cv2
import time
import csv
import os
from datetime import datetime
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Fatigue threshold
EYE_AR_THRESH = 0.23
CONSEC_FRAMES = 15

# Logging
log_path = "data/log.csv"
os.makedirs(os.path.dirname(log_path), exist_ok=True)

def log_event(status):
    with open(log_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status])

# Helper: Eye aspect ratio
def eye_aspect_ratio(landmarks, eye_indices, frame_w, frame_h):
    points = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in eye_indices]
    left = points[0]
    right = points[3]
    top = ((points[1][0] + points[2][0]) // 2, (points[1][1] + points[2][1]) // 2)
    bottom = ((points[4][0] + points[5][0]) // 2, (points[4][1] + points[5][1]) // 2)
    
    hor_dist = ((right[0] - left[0]) ** 2 + (right[1] - left[1]) ** 2) ** 0.5
    ver_dist = ((top[0] - bottom[0]) ** 2 + (top[1] - bottom[1]) ** 2) ** 0.5
    return ver_dist / hor_dist if hor_dist != 0 else 0

# Start video
cap = cv2.VideoCapture(0)
fatigue_counter = 0
fatigue_status = "Alert"

print("✅ Camera started. Fatigue detection in progress...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2

        if avg_ear < EYE_AR_THRESH:
            fatigue_counter += 1
        else:
            fatigue_counter = 0

        if fatigue_counter >= CONSEC_FRAMES:
            fatigue_status = "Fatigued"
        else:
            fatigue_status = "Alert"

        # Draw status on frame
        cv2.putText(frame, f"Status: {fatigue_status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if fatigue_status == "Fatigued" else (0, 255, 0), 2)

        # Log every 30 frames (~1 second)
        if int(time.time()) % 1 == 0:
            log_event(fatigue_status)

    cv2.imshow("NeuroCam - Fatigue Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
