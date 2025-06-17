
# import cv2
# import mediapipe as mp
# import time
# from scipy.spatial import distance as dist

# # MediaPipe Face Mesh setup
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
# mp_drawing = mp.solutions.drawing_utils

# # Eye landmark indices (from MediaPipe documentation)
# LEFT_EYE = [362, 385, 387, 263, 373, 380]
# RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# # Calculate Eye Aspect Ratio
# def eye_aspect_ratio(eye):
#     A = dist.euclidean(eye[1], eye[5])  # vertical
#     B = dist.euclidean(eye[2], eye[4])  # vertical
#     C = dist.euclidean(eye[0], eye[3])  # horizontal
#     return (A + B) / (2.0 * C)

# # Fatigue thresholds
# EAR_THRESHOLD = 0.25
# CONSEC_FRAMES = 20

# COUNTER = 0
# ALARM_ON = False

# # Start video stream
# cap = cv2.VideoCapture(0)

# print("🚨 Fatigue detection started...")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             h, w, _ = frame.shape

#             # Extract eye coordinates
#             left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
#             right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

#             # EAR computation
#             left_ear = eye_aspect_ratio(left_eye)
#             right_ear = eye_aspect_ratio(right_eye)
#             ear = (left_ear + right_ear) / 2.0

#             # Visualize eyes
#             for (x, y) in left_eye + right_eye:
#                 cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

#             # Check if EAR below threshold
#             if ear < EAR_THRESHOLD:
#                 COUNTER += 1
#                 if COUNTER >= CONSEC_FRAMES:
#                     cv2.putText(frame, "⚠️ DROWSINESS DETECTED!", (50, 100),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
#             else:
#                 COUNTER = 0

#             # Optional: show EAR value
#             cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     cv2.imshow("NeuroCam - Fatigue Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# app/fatigue_detector.py

import cv2
import mediapipe as mp
import time

# Initialize Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Eye landmark indices (right & left eye)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

def get_ear(eye_landmarks):
    from math import dist
    A = dist(eye_landmarks[1], eye_landmarks[5])
    B = dist(eye_landmarks[2], eye_landmarks[4])
    C = dist(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_fatigue(frame, results):
    fatigue = False
    h, w, _ = frame.shape
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]
            left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
            right_ear = get_ear(right_eye)
            left_ear = get_ear(left_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < 0.22:
                fatigue = True
    return fatigue
