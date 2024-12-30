import cv2
import numpy as np
from datetime import datetime
import os

# Initialize some variables
attendance_log = "attendance_log.csv"

# Ensure the attendance log file exists
if not os.path.exists(attendance_log):
    with open(attendance_log, "w") as f:
        f.write("Name,Timestamp\n")

# Load pre-trained face detector from OpenCV (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load images and labels
images = ["photos/Raheem.jpg", "photos/Ramesh.jpg", "photos/Vamshi.jpg"]
names = ["Raheem", "Ramesh", "Vamshi"]

face_encodings_list = []
for image_path in images:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        cropped_face = gray[y:y+h, x:x+w]
        resized_face = cv2.resize(cropped_face, (100, 100))  # Normalize size
        face_encodings_list.append(resized_face)

face_names_list = names

# Function to match faces
def match_face(input_face, face_encodings_list):
    input_gray = cv2.cvtColor(input_face, cv2.COLOR_BGR2GRAY)
    resized_input = cv2.resize(input_gray, (100, 100))  # Normalize size
    min_distance = float("inf")
    best_match = "Unknown"

    for idx, known_face in enumerate(face_encodings_list):
        distance = np.mean((known_face - resized_input) ** 2)
        print(f"Comparing with {face_names_list[idx]}, Distance: {distance}")
        if distance < min_distance:
            min_distance = distance
            best_match = face_names_list[idx]

    if min_distance < 1500:  # Fine-tuned threshold
        return best_match
    return "Unknown"

# Function to log attendance
def mark_attendance(name):
    with open(attendance_log, "a+") as f:
        f.seek(0)
        lines = f.readlines()
        logged_names = [line.split(",")[0] for line in lines]
        if name not in logged_names:
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{name},{timestamp}\n")

# Initialize webcam
video_capture = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame. Exiting.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        name = match_face(face, face_encodings_list)
        mark_attendance(name)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    frame_count += 1
    if frame_count % 10 == 0:
        cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
