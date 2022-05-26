import speech_recognition as sr  
import time

r = sr.Recognizer()

def listener():
    with sr.Microphone() as source:
        print("Speak:")
        aud = r.listen(source,phrase_time_limit=5)

    response = r.recognize_google(aud)
    return (response)