# face_recognition_project

# FaceTrack: Face Recognition Attendance System

## Requirements

- Python 3.x
- All dependencies in [requirements.txt](requirements.txt)
- Webcam

## Setup Instructions

### 1. Install Dependencies

```sh
pip install -r requirements.txt
```

### 2. Add Faces to Dataset

Run the following command and follow on-screen instructions to capture face data:

```sh
python add_faces.py
```

### 3. Train the Classifier

After adding faces, train the model:

```sh
python train_classifier.py
```

This will generate `classifier.pkl` and `label_encoder.pkl`.

### 4. Start the Application

Run the Flask app:

```sh
python app.py
```

The app will be available at [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

### 5. Using the Application

- **Home Page:** Shows the live video feed and recognizes faces.
- **Mark Attendance:** Attendance is marked by clicking on "Mark Attandance" button for recognized faces.
- **View Logs:** Click on "View Logs" to see attendance records by date.

### 6. Attendance Logs

Attendance records are saved as CSV files in the `Attendance/` directory. You can view them from the web interface or directly in the folder.

---

**Note:**  
If you add new faces, retrain the classifier using `train_classifier.py`.
