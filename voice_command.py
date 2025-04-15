import speech_recognition as sr

def listen_for_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for command...")
        audio = r.listen(source)
        try:
            command = r.recognize_google(audio)
            print(f"Command: {command}")
            return command
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError:
            print("Error with the speech recognition service")
            return None
