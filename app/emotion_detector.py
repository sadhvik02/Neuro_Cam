from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("app/best.pt")  # adjust path if needed

def detect_emotion(frame):
    results = model(frame, verbose=False)

    # Make the annotated image writable
    annotated_frame = results[0].plot().copy()  # ✅ <-- KEY FIX
    emotions = []

    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]
            emotions.append(label)

    return annotated_frame, emotions

