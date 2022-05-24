#!/usr/bin/env python3                                                                                

from time import time
import speech_recognition as sr  
import os
import playsound
from gtts import gTTS



def speak(text):
    tts = gTTS(text = text, lang = "en")
    filename = "voice.mp3"
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)

# get audio from the microphone                                                                       
r = sr.Recognizer()

for i in range(2):
    with sr.Microphone() as source:                                                                       
        print("Speak:")                                                                                   
        audio = r.listen(source)   

    try:
        speak("Did you say" + r.recognize_google(audio))
        print("You said " + r.recognize_google(audio))
    except sr.UnknownValueError:
        print("Could not understand audio")
        speak("Could not understand audio")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        speak("Could not request results")


