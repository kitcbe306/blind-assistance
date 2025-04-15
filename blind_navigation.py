from flask import Flask, render_template, Response
import cv2
import torch
from gtts import gTTS
import os
import platform
import threading
import speech_recognition as sr

app = Flask(__name__)

# Global variables
camera_running = False
cap = None

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Function to generate a beep sound
def beep():
    system_platform = platform.system()
    if system_platform == "Windows":
        import winsound
        winsound.Beep(1000, 500)  # Frequency: 1000Hz, Duration: 500ms
    else:
        os.system('echo -e "\a"')  # Linux/Mac

# Function to speak object names
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("object.mp3")
    os.system("mpg321 object.mp3")  # Use 'mpg321' or any audio player

# Function to listen for speech commands
def listen_for_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for command...")
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio).lower()
        print(f"Command recognized: {command}")
        return command
    except sr.UnknownValueError:
        print("Could not understand the audio.")
        return None
    except sr.RequestError:
        print("Could not request results from the speech recognition service.")
        return None

# Function to identify the object in front
def identify_object_in_front(frame):
    # Perform object detection
    results = model(frame)
    detections = results.xyxy[0].numpy()

    # Find the object with the largest bounding box (closest object)
    largest_area = 0
    closest_object = None

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        if conf > 0.5:  # Confidence threshold
            label = model.names[int(cls)]
            area = (x2 - x1) * (y2 - y1)  # Calculate bounding box area
            if area > largest_area:
                largest_area = area
                closest_object = label

    return closest_object

# Function to run the navigation system
def run_navigation():
    global camera_running, cap
    cap = cv2.VideoCapture(0)
    while camera_running:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)
        detections = results.xyxy[0].numpy()

        # Check for obstacles and speak object names
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if conf > 0.5:  # Confidence threshold
                label = model.names[int(cls)]
                print(f"Detected: {label}")

                # Calculate bounding box height
                bbox_height = y2 - y1

                # Define thresholds for 2 to 4 feet
                # Adjust these values based on your camera and environment
                MIN_HEIGHT = 100  # Minimum height for 4 feet
                MAX_HEIGHT = 200  # Maximum height for 2 feet

                # Check if the object is within 2 to 4 feet
                if MIN_HEIGHT <= bbox_height <= MAX_HEIGHT:
                    print(f"Beep! {label} detected within 2 to 4 feet.")
                    beep()  # Call the beep function
                else:
                    print(f"{label} detected, but outside 2 to 4 feet range.")

        # Listen for the "What is in front of me?" command
        command = listen_for_command()
        if command and "what is in front of me" in command:
            closest_object = identify_object_in_front(frame)
            if closest_object:
                speak(f"There is a {closest_object} in front of you.")
            else:
                speak("I don't see any object in front of you.")

    # Release the camera when stopped
    if cap:
        cap.release()
    print("Navigation stopped.")

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to start the navigation
@app.route('/start')
def start():
    global camera_running
    if not camera_running:
        camera_running = True
        threading.Thread(target=run_navigation).start()
        return "Navigation started!"
    return "Navigation is already running!"

# Route to stop the navigation
@app.route('/stop')
def stop():
    global camera_running
    if camera_running:
        camera_running = False
        return "Navigation stopped!"
    return "Navigation is not running!"

if __name__ == '__main__':
    app.run(debug=True)