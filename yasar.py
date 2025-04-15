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
        self.voice_thread = threading.Thread(target=self.beep_feedback, daemon=True)
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
            self.provide_feedback(is_very_close=True)  # Play warning beep if the camera is not accessible
            return

        while navigation_running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_resized = cv2.resize(frame, (640, 480))
            objects = detect_objects(frame_resized)

            for obj_name, conf, center_x, center_y in objects:
                if conf > 0.5:  # Only consider objects with confidence > 50%
                    distance = estimate_distance(center_x, center_y, frame_resized.shape[1], frame_resized.shape[0])

                    if distance < 1.0:
                        self.provide_feedback(is_very_close=True)  # Beep 5 times for close obstacles
                    else:
                        self.provide_feedback(is_very_close=False)  # Single beep for other objects

            # Display the video feed in the Kivy window
            self.display_video(frame_resized)

            time.sleep(0.1)

    def display_video(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(frame_rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        self.image_widget.texture = texture

    def provide_feedback(self, is_very_close=False):
        sound = SoundLoader.load('alert_sound.wav')
        if not sound:
            print("Error: alert_sound.wav not found.")
            return

        def beep_feedback():
            if is_very_close:
                for _ in range(5):  # Beep 5 times for very close obstacles
                    sound.play()
                    time.sleep(0.6)  # 0.6-second interval
            else:
                sound.play()  # Single beep for detected objects

        threading.Thread(target=beep_feedback, daemon=True).start()


class MainApp(App):
    def build(self):
        return NavigationApp()


if __name__ == "__main__":
    MainApp().run()
