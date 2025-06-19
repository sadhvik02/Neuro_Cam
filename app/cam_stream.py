"""cam_stream_emotion_fatigue.py
Combine YOLOv8 emotion detection (best.pt) with MediaPipe-based fatigue detection.
Press 'q' to quit the stream.
Dependencies: ultralytics, mediapipe, opencv-python
"""

import cv2
import csv
import os
from datetime import datetime
import mediapipe as mp
from emotion_detector import detect_emotion  # ensure emotion_detector.py is in PYTHONPATH and loads best.pt

# --------------------- Fatigue‑detection setup --------------------- #
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Landmark indices for eye aspect ratio
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EYE_AR_THRESH = 0.23  # Empirical threshold for eye aspect ratio
CONSEC_FRAMES = 15    # Number of consecutive frames below thresh ➜ fatigued

# Logging setup
log_path = "data/log.csv"
os.makedirs(os.path.dirname(log_path), exist_ok=True)

def log_event(event_type: str, value: str):
    """Append an event (emotion or fatigue) to CSV with timestamp."""
    with open(log_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event_type, value])


def eye_aspect_ratio(landmarks, eye_indices, w: int, h: int) -> float:
    """Compute eye aspect ratio for a given eye."""
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    left, right = pts[0], pts[3]
    top = ((pts[1][0] + pts[2][0]) // 2, (pts[1][1] + pts[2][1]) // 2)
    bottom = ((pts[4][0] + pts[5][0]) // 2, (pts[4][1] + pts[5][1]) // 2)
    hor = ((right[0] - left[0]) ** 2 + (right[1] - left[1]) ** 2) ** 0.5
    ver = ((top[0] - bottom[0]) ** 2 + (top[1] - bottom[1]) ** 2) ** 0.5
    return ver / hor if hor else 0.0


# --------------------- Main camera loop --------------------- #
cap = cv2.VideoCapture(0)
fatigue_counter = 0
prev_fatigue_status = "Alert"

print("✅ Camera started. Emotion + Fatigue detection running… (press 'q' to quit)")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # ---------- Emotion detection (YOLOv8) ---------- #
    out_frame, emotions = detect_emotion(frame)
    emotion_label = emotions[0] if emotions else "None"

    # Overlay emotion text
    cv2.putText(out_frame, f"Emotion: {emotion_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Log new emotion (on change)
    if hasattr(log_event, "last_emotion"):
        if emotion_label != log_event.last_emotion:
            log_event("emotion", emotion_label)
            log_event.last_emotion = emotion_label
    else:
        log_event("emotion", emotion_label)
        log_event.last_emotion = emotion_label

    # ---------- Fatigue detection (MediaPipe) ---------- #
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        ear = (left_ear + right_ear) / 2.0

        if ear < EYE_AR_THRESH:
            fatigue_counter += 1
        else:
            fatigue_counter = 0

        fatigue_status = "Fatigued" if fatigue_counter >= CONSEC_FRAMES else "Alert"
    else:
        fatigue_status = "No face"

    # Overlay fatigue status
    color = (0, 0, 255) if fatigue_status == "Fatigued" else (0, 255, 0)
    cv2.putText(out_frame, f"Fatigue: {fatigue_status}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Log fatigue status on change
    if fatigue_status != prev_fatigue_status:
        log_event("fatigue", fatigue_status)
        prev_fatigue_status = fatigue_status

    # ---------- Display ---------- #
    cv2.imshow("NeuroCam – Emotion & Fatigue", out_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()