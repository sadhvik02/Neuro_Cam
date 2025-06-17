import cv2
import csv
import json
from datetime import datetime
import os

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(data_dir, exist_ok=True)

log_file = os.path.join(data_dir, 'log.csv')
session_file = os.path.join(data_dir, 'session.json')

def log_event(event_type, details):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([now, event_type, details])

def save_session(data):
    with open(session_file, 'w') as f:
        json.dump(data, f, indent=2)

def draw_bbox_with_label(img, bbox, label, color=(0, 255, 0)):
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
