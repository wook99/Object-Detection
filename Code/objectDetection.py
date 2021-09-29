import cv2
import numpy as np
import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime
import wikipedia
import pyjokes
from tkinter import *
import tkinter.font as font

from PIL import ImageTk ,Image
from tkinter import messagebox

window = Tk()
window.title('Object Detection App')
window.geometry("800x455")
#window['background']='#856ff8'

background_image=PhotoImage(file="Presentation1.png")
background_label = Label(window, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)


myImg = ImageTk.PhotoImage(Image.open("ms3.png"))
myLabel = Label(image=myImg)
myLabel.place(x=300,y=100)
# Resizing image to fit on button 
#photoimage = photo.subsample(3, 3) 
  
# here, image option is used to 
# set image on button 
# compound option is used to align 
# image on LEFT side of button 

listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
command=""



thres = 0.7 # Threshold to detect object
nms_threshold = 0.2
cap = cv2.VideoCapture(1)
# cap.set(3,1280)
# cap.set(4,720)
# cap.set(10,150)
classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#print(classNames)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)




def talk(text):
    engine.say(text)
    engine.runAndWait()



def take_command():
    command=""
    try:
        with sr.Microphone() as source:
            print('listening...')
            voice = listener.listen(source)
            command = listener.recognize_google(voice)
            command = command.lower()
            if 'alexa' in command:
                command = command.replace('alexa', '')
                print(command)
    except:
        pass
    return command


def run_alexa(command2):
    command2=take_command()
    print(command2)
    if 'no' in command2:
         talk("Ok good bye have a nice day")
         exit(0)
    if 'hello' in command2:
        talk("Hello")
        run_alexa("hello")
    if 'play' in command2:
        song = command2.replace('play', '')
        talk('playing ' + song)
        pywhatkit.playonyt(song)
        talk("Is there anything else you need to know")
        run_alexa("")
    if 'time' in command2:
        time = datetime.datetime.now().strftime('%I:%M %p')
        talk('Current time is ' + time)
        talk("Is there anything else you need to know")
        run_alexa("")
    if 'who is' in command2:
        try:
            person = command2.replace('who is', '')
            info = wikipedia.summary(person, 1)
            print(info)
            talk(info)
        except:
            talk("Sorry couldn't find any match")
        
        talk("Is there anything else you need to know")
        run_alexa("")
    if 'date' in command2:
        talk('sorry, I have a headache')
        talk("Is there anything else you need to know")
        run_alexa("")
    if 'are you single' in command2:
        talk('I am in a relationship with wifi')
        talk("Is there anything else you need to know")
        run_alexa("")
    if 'joke' in command2:
        talk(pyjokes.get_joke())
        talk("Is there anything else you need to know")
        run_alexa("")
    if 'recognise' in command2:
        talk('ok.Sure')
        mm()
    if 'yes' in command2:
        talk("what is it")
        run_alexa("hello")
    else:
        talk('Please say the command again.')
        run_alexa("")


def mm():
    while True:
        success,img = cap.read()
        classIds, confs, bbox = net.detect(img,confThreshold=thres)
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1,-1)[0])
        confs = list(map(float,confs))
        #print(type(confs[0]))
        #print(confs)

        indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
        #print(indices)

        for i in indices:
            i = i[0]
            box = bbox[i]
            x,y,w,h = box[0],box[1],box[2],box[3]
            cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
            cv2.putText(img,classNames[classIds[i][0]-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            if cv2.waitKey(1) & 0xFF == ord('p'):
                talk("It is a "+classNames[classIds[i][0]-1])

        cv2.imshow("Output",img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            talk("Do you want to continue recognisiing objects?")
            command3=take_command()
            print(command3)
            if "yes" in command3:
                mm()
            elif 'no' in command3:                
                talk("Is there anything else you need to know")
                run_alexa("")
            else:
                talk("Sorry i dont understand Can you please tell me again")
                run_alexa("")

def addNew():
    talk("Hello how can I help you")
    run_alexa("hello")

#myButton1(window, text = 'Click Me !', image = photoimage, 
                    #compound = LEFT,command=addNew).pack(side = TOP) 
myFont = font.Font(size=14)
myButton1 = Button(window, text="click me!",bg='#0052cc', fg='#ffffff', command=addNew)
myButton1['font'] = myFont
myButton1.place(x=335,y=290)
window.mainloop()

cap.release()
cv2.destroyAllWindows()
