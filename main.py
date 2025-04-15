import threading
import time
import cv2
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.core.audio import SoundLoader
import pyttsx3  # for text-to-speech
from object_detection import detect_objects  # This function will use YOLOv5
from voice_command import listen_for_command  # This will handle voice commands
from haptic_feedback import trigger_haptic_feedback  # Handle feedback when obstacles are close

# Initialize the navigation state
navigation_running = False

# Simple distance estimation for object proximity (using a hypothetical conversion factor to meters)
def estimate_distance(center_x, center_y, frame_width, frame_height, conversion_factor=0.05):
    # Using a basic proportional method based on frame center and a conversion factor for meters
    distance_pixels = ((frame_width / 2 - center_x) ** 2 + (frame_height / 2 - center_y) ** 2) ** 0.5
    distance_meters = distance_pixels * conversion_factor
    return distance_meters


class NavigationApp(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'

        # Create the top section with the status label
        self.status_label = Label(text="Ready to start navigation.", size_hint=(1, 0.1))
        self.add_widget(self.status_label)

        # Create video feed section
        self.image_widget = Image(size_hint=(1, 0.7))
        self.add_widget(self.image_widget)

        # Create the bottom section with control buttons
        control_layout = BoxLayout(size_hint=(1, 0.2), orientation='horizontal')

        # Add Start and Stop buttons (for manual control)
        self.start_button = Button(text="Start Navigation", on_press=self.start_navigation)
        self.stop_button = Button(text="Stop Navigation", on_press=self.stop_navigation)
        control_layout.add_widget(self.start_button)
        control_layout.add_widget(self.stop_button)

        self.add_widget(control_layout)

        # Start voice command listener in a separate thread
        self.voice_thread = threading.Thread(target=self.listen_for_voice_commands, daemon=True)
        self.voice_thread.start()

    def start_navigation(self, instance):
        if self.status_label:
            self.status_label.text = "Navigation Started"
        global navigation_running
        navigation_running = True
        threading.Thread(target=self.run_navigation, daemon=True).start()

    def stop_navigation(self, instance):
        if self.status_label:
            self.status_label.text = "Navigation Stopped"
        global navigation_running
        navigation_running = False

    def run_navigation(self):
        cap = cv2.VideoCapture(0)  # Open the camera
        if not cap.isOpened():
            self.provide_feedback("Camera not accessible!")
            return  # Exit if the camera is not available

        while navigation_running:
            ret, frame = cap.read()
            if not ret:
                self.provide_feedback("Error in video feed!")
                continue  # Skip the current iteration and try again

            # Resize frame for efficient processing
            frame_resized = cv2.resize(frame, (640, 480))

            # Detect objects using YOLOv5
            objects = detect_objects(frame_resized)

            # Process objects and give feedback for each detected object
            detected_objects = []
            for obj_name, conf, center_x, center_y in objects:
                if conf > 0.5:  # If confidence is above 50%
                    detected_objects.append(obj_name)
                    # Calculate distance from the center of the frame in meters
                    distance = estimate_distance(center_x, center_y, frame_resized.shape[1], frame_resized.shape[0])
                    
                    if distance < 1.0:  # If the obstacle is close (within 1 meter)
                        self.provide_feedback(f"Warning! {obj_name} is very close, less than 1 meter.")
                        trigger_haptic_feedback()  # Trigger haptic feedback
                    else:
                        self.provide_feedback(f"{obj_name} detected at a distance of {distance:.2f} meters.")

            # Provide feedback for all detected objects
            if detected_objects:
                self.provide_feedback(f"Objects detected: {', '.join(detected_objects)}.")
            else:
                self.provide_feedback("No obstacles detected.")

            # Display the video feed in the Kivy window
            self.display_video(frame_resized)

            time.sleep(0.1)  # Small delay between frames

    def display_video(self, frame):
        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the frame to texture and update the image widget
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(frame_rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        self.image_widget.texture = texture

    def provide_feedback(self, text):
        # Text-to-Speech feedback
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

        # Optional: Play sound as additional feedback
        sound = SoundLoader.load('alert_sound.wav')  # Load sound for feedback
        if sound:
            sound.play()

    def listen_for_voice_commands(self):
        # Continuously listen for voice commands in a separate thread
        while True:
            command = listen_for_command()  # Call the function to listen for voice commands
            if command:
                # If the command is "detect", provide feedback about the obstacles
                if 'detect' in command.lower() and navigation_running:
                    self.provide_feedback("Detecting obstacles...")
                    # You can also trigger a check to call detect_objects manually if needed


class MainApp(App):
    def build(self):
        return NavigationApp()


if __name__ == "__main__":
    MainApp().run()
