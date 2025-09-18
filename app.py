import os
import cv2
import pickle
import time
import csv
from datetime import datetime
from flask import Flask, render_template, Response, request
from win32com.client import Dispatch
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

app = Flask(__name__)

# Load THE TRAINED CLASSIFIER AND MODELS 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load face detector and recognition models
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load the trained SVM classifier and label encoder
try:
    with open('classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print("Classifier and label encoder loaded successfully.")
except FileNotFoundError:
    print("FATAL ERROR: classifier.pkl or label_encoder.pkl not found.")
    print("Please run train_classifier.py first to generate these files.")
    exit()

last_recognized_name = None

# FRAME GENERATOR WITH ML CLASSIFIER ---
def gen_frames():
    global last_recognized_name
    """Captures frames from webcam and performs face recognition."""
    video = cv2.VideoCapture(0)

    while True:
        ret, frame = video.read()
        if not ret or frame is None:
            print("Failed to grab frame")
            continue

        # Detect faces
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb_frame)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]

                # Ensure box dimensions are valid
                if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                    continue

                face_img = rgb_frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue

                # Get face embedding
                face_img = cv2.resize(face_img, (160, 160))
                face_tensor = torch.tensor(face_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                face_tensor = face_tensor.to(device)
                embedding = resnet(face_tensor).detach().cpu().numpy()

                # Use the Classifier for Prediction ---
                predictions = classifier.predict_proba(embedding)[0]
                best_class_index = np.argmax(predictions)
                best_class_probability = predictions[best_class_index]
                
                confidence_threshold = 0.85  # Confidence level for a match

                name = "Unknown"
                if best_class_probability > confidence_threshold:
                    name = label_encoder.inverse_transform([best_class_index])[0]

                # Draw bounding box and display name with confidence
                if name == "Unknown":
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, name, (x1, y1 - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                else:
                    display_text = f"{name} ({best_class_probability*100:.0f}%)"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, display_text, (x1, y1 - 15), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

                if name != "Unknown":
                    last_recognized_name = name

        # Encode image for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    video.release()

# FLASK ROUTES
@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/video')
def video_feed():
    """Provides the video stream."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs', methods=['GET', 'POST'])
def view_logs():
    """Displays attendance logs."""
    logs_dir = 'Attendance'
    os.makedirs(logs_dir, exist_ok=True) # Ensure directory exists
    files = sorted([f for f in os.listdir(logs_dir) if f.endswith('.csv')], reverse=True)
    
    selected_file = request.form.get('date') if request.method == 'POST' else (files[0] if files else None)
    attendance_data = []

    if selected_file and os.path.exists(os.path.join(logs_dir, selected_file)):
        with open(os.path.join(logs_dir, selected_file), 'r') as f:
            reader = csv.reader(f)
            try:
                header = next(reader) # Skip header
                attendance_data = list(reader)
            except StopIteration:
                attendance_data = [] # File is empty

    return render_template('logs.html', files=files, attendance_data=attendance_data, selected_file=selected_file)

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    global last_recognized_name
    if not last_recognized_name or last_recognized_name == "Unknown":
        return {"status": "fail", "message": "No recognized face"}, 400

    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
    timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
    filename = f"Attendance/Attendance_{date}.csv"
    attendance_entry = [str(last_recognized_name), str(timestamp)]

    os.makedirs("Attendance", exist_ok=True)
    already_marked = False
    if os.path.isfile(filename):
        with open(filename, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Skip header row
            for row in reader:
                if row and row[0] == last_recognized_name:
                    already_marked = True
                    break

    if not already_marked:
        with open(filename, "a", newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['NAME', 'TIME'])
            writer.writerow(attendance_entry)
        return {"status": "success", "message": f"Attendance marked for {last_recognized_name}"}
    else:
        return {"status": "fail", "message": "Already marked"}, 200

if __name__ == '__main__':
    app.run(debug=True)