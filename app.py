from flask import Flask, render_template, jsonify, Response
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
show_camera = False

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

# Function to speak object names (cross-platform)
def speak(text):
    try:
        if os.path.exists("object.mp3"):
            os.remove("object.mp3")

        tts = gTTS(text=text, lang='en')
        tts.save("object.mp3")

        if platform.system() == "Windows":
            from playsound import playsound
            playsound("object.mp3")
        else:
            os.system("mpg321 object.mp3")

        if os.path.exists("object.mp3"):
            os.remove("object.mp3")
    except Exception as e:
        print(f"Error in speak(): {str(e)}")

# Function to identify all objects in the defined path
def identify_object_in_path(frame):
    try:
        # Define the path (middle 20% width and 80% height of the frame)
        height, width, _ = frame.shape
        path_x1 = int(width * 0.4)  # 20% from left
        path_x2 = int(width * 0.6)  # 20% from right
        path_y1 = int(height * 0.1)  # 10% from top
        path_y2 = int(height * 0.9)  # 10% from bottom

        # Draw the path on the frame
        cv2.rectangle(frame, (path_x1, path_y1), (path_x2, path_y2), (0, 255, 0), 2)

        # Crop the frame to the path region
        path_frame = frame[path_y1:path_y2, path_x1:path_x2]

        # Perform object detection on the path region
        results = model(path_frame)
        detections = results.xyxy[0].numpy()

        detected_objects = []

        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if conf > 0.5:  # Confidence threshold
                label = model.names[int(cls)]
                detected_objects.append(label)
                print(f"Detected: {label} (Confidence: {conf:.2f})")

        return detected_objects
    except Exception as e:
        print(f"Error in identify_object_in_path(): {str(e)}")
        return None

# Function to run navigation system
def run_navigation():
    global camera_running, cap, show_camera
    cap = cv2.VideoCapture(0)
    while camera_running:
        ret, frame = cap.read()
        if not ret:
            break

        # Identify objects in the path
        detected_objects = identify_object_in_path(frame)

        # Trigger beep and speak if objects are detected
        if detected_objects:
            beep()
            speak(f"There is a {', '.join(detected_objects)} in front of you.")

        # Show the camera feed if enabled
        if show_camera:
            cv2.imshow("Camera Feed", frame)
            cv2.waitKey(1)

    if cap:
        cap.release()
    cv2.destroyAllWindows()
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

# Route to toggle camera view
@app.route('/toggle_camera')
def toggle_camera():
    global show_camera
    show_camera = not show_camera
    return f"Camera view {'enabled' if show_camera else 'disabled'}."

# Route to identify the object in front
@app.route('/identify_object')
def identify_object():
    global cap
    if cap is None or not cap.isOpened():
        return jsonify({"error": "Camera is not running."})

    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Failed to capture frame."})

    detected_objects = identify_object_in_path(frame)
    if detected_objects:
        speak(f"There is a {', '.join(detected_objects)} in front of you.")
        return jsonify({"objects": detected_objects})
    else:
        speak("I don't see any object.")
        return jsonify({"objects": None})

# Route to stream camera feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global cap, show_camera
    while show_camera:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw the path on the frame
        height, width, _ = frame.shape
        path_x1 = int(width * 0.4)
        path_x2 = int(width * 0.6)
        path_y1 = int(height * 0.1)
        path_y2 = int(height * 0.9)
        cv2.rectangle(frame, (path_x1, path_y1), (path_x2, path_y2), (0, 255, 0), 2)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True)