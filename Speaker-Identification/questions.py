from re import I
import pyttsx3
import requests
from snapshots import *
import speech_recognition as sr  
import playsound
from listen import *
from test import *

# initialisation
engine = pyttsx3.init()

def mcq_for_test():
    response = requests.get("https://my-json-server.typicode.com/aswinikalyan30/jsonserver/db")
    data = response.json()["questions"]
    import time


    c_answers = []
    answers = []


    for i in data:
        c_answers.append(str(i["correct_answer"]))

    i=0

    while(i<len(data)):
        #c_answers.append(str(data[i]["correct_answer"]))
        engine.say(str(data[i]["question"]))
        engine.say("A " + str(data[i]["answers"]["a"]))
        engine.say("B " + str(data[i]["answers"]["b"]))
        engine.say("C " + str(data[i]["answers"]["c"]))
        engine.say("D " + str(data[i]["answers"]["d"]))
        engine.say("..." + str(data[i]["answers"]["e"]))
        engine.say("Answer Now:")

        engine.runAndWait()
        try:
            print("before",i)
            ans = listener()
            snapshot()
            if(ans=='repeat'):
                print("after",i)
                engine.say("Question will be repeated. Please answer carefully")
                engine.runAndWait()
                continue
            
            engine.say("Is your answer option " + ans)
            engine.runAndWait()

            confirm = listener()

            engine.say("You said " + confirm)
            engine.runAndWait()
            if confirm.lower() == "no":
                engine.say("Question will be repeated. Please answer carefully")
                engine.runAndWait()
                continue

            answers.append(ans)
            print(ans)
        except:
            answers.append('a')
            print("a")
        if(i < 3):
            i+=1
        engine.say("Say YES to proceed ")
        engine.runAndWait()
        record_audio_test()
        test_model()
    score=0
    print(c_answers,answers,score)

    for j in range(len(c_answers)):
        if(c_answers[j]==answers[j]):
            score+=1

    print(c_answers,answers,score)

    engine.say("You have scored a total of " + str(score) +" out of "+str(len(data)) + " questions")
    engine.runAndWait()
