import cv2
import os

cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

print("🧪 Script started")

# Load image
img_path = os.path.join(os.path.dirname(__file__), "sample_face.jpg")
img = cv2.imread(img_path)

if img is None:
    print(f"❌ Image not found at: {img_path}")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

print(f"✅ Detected {len(faces)} face(s)")

# Optional: draw and show the image
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Detected Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
