import cv2
import numpy as np
from keras.models import load_model
import pygame
import time
import pandas as pd
import csv
from datetime import date
from flask import Flask, render_template, Response, jsonify, send_file, request

# Load the trained models and cascade classifiers
# (Make sure to update the paths to your model and cascade classifier files)
student_model = load_model("../model/students_face.h5")#students_face.h5
model = load_model('../model/eyes_state.h5')
emotion_model = load_model("../model/emotions.h5")
face_cascade = cv2.CascadeClassifier('../model/haarcascade_frontalface_default.xml')

# Define the labels for eye states
eye_labels = {
    0: "Open",
    1: "Closed"
}

# CSV file details
csv_filename = "../csv_data/student_data.csv"
csv_headers = ["Date", "Student ID", "Student Name", "Stress Percentage"]
# Check if CSV file exists
file_exists = False
try:
    with open(csv_filename, 'r'):
        file_exists = True
except FileNotFoundError:
    pass
# Function to preprocess the eye region
def preprocess_eye(eye):
    resized_eye = cv2.resize(eye, (48, 48))
    normalized_eye = resized_eye / 255.0
    preprocessed_eye = np.expand_dims(normalized_eye, axis=-1)
    return preprocessed_eye

# Load the student IDs and corresponding labels
student_ids = {}
with open('../csv_data/students.csv', 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        student_id = int(row[0])
        student_name = row[1]
        student_ids[student_id] = student_name

# Define stress emotions
stress_emotions = [0, 1, 2, 5]  # Angry, Disgust, Fear, Sad

# Initialize variables for stress calculation
total_frames = 0
stress_frames = 0
is_call_ended = False

# Open the webcam
cap = cv2.VideoCapture(0)

# Initialize variables for tracking eye closure duration
closed_start_time = None
closed_eyes_duration = 0
closed_eyes_threshold = 1  # Number of consecutive frames to consider eyes closed
is_closed_eyes_sound_playing = False
is_sound_playing = False

# Initialize Flask app
app = Flask(__name__)
#app = Flask(__name__, static_url_path='/static')

def process_frame(frame):
    global total_frames, stress_frames, closed_start_time, is_sound_playing, closed_eyes_duration, is_closed_eyes_sound_playing

    # Initialize sys_state with an empty string
    sys_state = ""
    predicted_emotion = "Neutral"

    # Detect and recognize student faces in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if faces are detected
    if len(faces) == 0:
        # If no faces are detected, play the alert sound
        if not is_sound_playing:
            pygame.mixer.init()
            pygame.mixer.music.load('../alert/face_not_detected_sound.mp3')
            pygame.mixer.music.play(-1)  # Play the alert sound repeatedly
            is_sound_playing = True
    else:
        # Stop the alert sound if it is playing and reset the flag
        if is_sound_playing:
            pygame.mixer.music.stop()
            is_sound_playing = False

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI)
        face_roi = gray[y:y + h, x:x + w]

        # Resize the face ROI to match the input size of the model
        face_roi = cv2.resize(face_roi, (48, 48))

        # Normalize the face ROI
        face_roi = face_roi / 255.0

        # Reshape the face ROI to match the input shape of the model
        face_roi = np.reshape(face_roi, (1, 48, 48, 1))

        # Perform prediction using the model for student face recognition
        student_predictions = student_model.predict(face_roi)
        student_predicted_label = np.argmax(student_predictions[0])

        # Get the corresponding student ID
        student_id = student_predicted_label

        # Perform prediction using the model for emotion detection
        emotion_predictions = emotion_model.predict(face_roi)
        emotion_index = np.argmax(emotion_predictions[0])

        # Calculate stress frames
        if emotion_index in stress_emotions:
            stress_frames += 1

        total_frames += 1

        # Get the predicted emotion label
        emotion_labels = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Neutral",
            5: "Sad",
            6: "Surprise"
        }
        predicted_emotion = emotion_labels[emotion_index]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get the corresponding student name
        student_name = student_ids.get(student_id, "Unknown")

        # Display the student ID and predicted emotion label
        cv2.putText(frame, "Student Name: {}".format(student_name), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)
        cv2.putText(frame, "Emotion: {}".format(predicted_emotion), (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

        # Preprocess the frame to detect eyes
        eyes = cv2.CascadeClassifier('../model/haarcascade_eye.xml').detectMultiScale(gray)

        # If eyes are detected
        if len(eyes) > 0:
            # Combine the eye regions
            combined_eye_roi = np.zeros((48, 48), dtype=np.uint8)

            # Process each detected eye
            for (x, y, w, h) in eyes:
                # Extract the eye region
                eye_roi = gray[y:y + h, x:x + w]

                # Preprocess the eye region
                preprocessed_eye = preprocess_eye(eye_roi)

                # Add the preprocessed eye to the combined eye region
                combined_eye_roi = np.maximum(combined_eye_roi, preprocessed_eye.squeeze())

            # Perform prediction on the combined eye region
            prediction = model.predict(np.expand_dims(combined_eye_roi, axis=0))
            predicted_class = eye_labels[prediction.argmax()]

            # Display the eye state in the frame
            cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Check if eyes are closed
            if predicted_class == "Closed":
                if closed_start_time is None:
                    closed_start_time = time.time()
                else:
                    # Calculate the duration of closed eyes
                    closed_eyes_duration = time.time() - closed_start_time

                    # Check if eyes are closed for more than the threshold
                    if closed_eyes_duration > closed_eyes_threshold and not is_closed_eyes_sound_playing:
                        # Play the sound and repeat until eyes are open
                        pygame.mixer.init()
                        pygame.mixer.music.load('../alert/sleeping_sound.mp3')
                        pygame.mixer.music.play(-1)
                        is_closed_eyes_sound_playing = True
            else:
                # Eyes are open, stop the sound if it is playing
                if is_closed_eyes_sound_playing:
                    pygame.mixer.music.stop()
                    is_closed_eyes_sound_playing = False
                closed_start_time = None
                closed_eyes_duration = 0
        else:
            # Reset predicted_emotion to "Neutral" if no eyes are detected
            predicted_emotion = "Neutral"

    # Calculate stress percentage only when the call is not ended
    if not is_call_ended:
        stress_percentage = (stress_frames / total_frames) * 100 if total_frames > 0 else 0
        print("Stress Percentage:", stress_percentage)

        # Check if stress percentage is above 50%
        if stress_percentage > 50:
            sys_state = "System State: High Stress Level Detected!"
        else:
            sys_state = "System State: Normal"

    return frame, sys_state, "Emotion State: " + predicted_emotion

def generate_frames():
    while True:
        success, frame = cap.read()

        if not success:
            break

        frame, sys_state, emotion_state = process_frame(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    sys_state = ""
    emotion_state = ""
    return render_template('index.html', sys_state=sys_state, emotion_state=emotion_state)

# Route to serve the CSV file
@app.route('/download_csv')
def download_csv():
    # Replace 'static/csv_data/student_data.csv' with the correct path to your CSV file
    file_path = '../csv_data/student_data.csv'
    return send_file(file_path, as_attachment=True)

@app.route('/download_attendance_report')
def download_attendance_report():
    # Read the student data CSV file
    student_data = pd.read_csv("../csv_data/student_data.csv")

    # Convert the Date column to datetime format
    student_data["Date"] = pd.to_datetime(student_data["Date"])

    # Drop duplicate rows based on Student ID and Date
    student_data_unique = student_data.drop_duplicates(subset=["Student ID", "Date"])

    # Group the unique data by Student ID and calculate the attendance counts
    attendance_counts = student_data_unique.groupby(["Student ID", "Student Name"]).size().reset_index(
        name="Attendance Count")

    # Calculate total number of days
    total_days = student_data["Date"].nunique()

    # Calculate the number of absence days for each student
    attendance_counts["Absence Count"] = total_days - attendance_counts["Attendance Count"]

    # Save the attendance report DataFrame to a new CSV file
    attendance_report = attendance_counts[["Student ID", "Student Name", "Attendance Count", "Absence Count"]]
    attendance_report.to_csv("../csv_data/attendance_report.csv", index=False)
    file_path = '../csv_data/attendance_report.csv';

    print("Attendance report CSV file created successfully.")
    return send_file(file_path, as_attachment=True)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/end_call', methods=['POST'])
def end_call():
    global is_call_ended

    # Set the flag to indicate that the call has ended
    is_call_ended = True

    # Calculate stress percentage
    stress_percentage = (stress_frames / total_frames) * 100
    print("Stress Percentage:", stress_percentage)

    # Check if stress percentage is above 50%
    if stress_percentage > 50:
        print("High Stress Level Detected!")
        # Play audio file
        pygame.mixer.init()
        pygame.mixer.music.load('../alert/stress_sound.mp3')
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
        pygame.mixer.quit()

    # Save student data to CSV file
    data = [str(date.today()), student_id, student_ids[student_id], stress_percentage]

    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(csv_headers)  # Write headers if the file is newly created
        writer.writerow(data)

    print("Student data saved to CSV file:", csv_filename)

    # Reset stress calculation for the next call
    # global total_frames, stress_frames
    # total_frames = 0
    # stress_frames = 0

    # Add any necessary code for ending the call or further processing
    return jsonify({'message': 'Call ended successfully',
                    'stress_percentage': stress_percentage})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)

