from flask import Flask, render_template, Response
import cv2
import torch
from gtts import gTTS
import os
import platform
import threading

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

                # Obstacle warning (beep sound)
                if label in ['person', 'cell phone' , 'chair']:  # Add more objects as needed
                    print(f"Beep! {label} detected.")
                    beep()  # Call the beep function

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