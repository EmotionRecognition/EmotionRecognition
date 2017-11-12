import cv2
import random
import tkinter as tk
from tkinter import *

import time
from PIL import Image
from PIL import ImageTk

from enum import Enum

# Variables
percentValues = {
    'smutek': 0,
    'zaskoczenie': 0,
    'strach': 0,
    'podziw': 0,
    'radosc': 0,
    'zainteresowanie': 0,
    'spokoj': 0,
    'obrzydzenie': 0
}

class Emotions(Enum):
   smutek = 1
   zaskoczenie = 2
   strach = 3
   podziw = 4
   radosc = 5
   zainteresowanie = 6
   spokoj = 7
   obrzydzenie = 8


# setting for Thinker gui
window = tk.Tk()
window.wm_title("Camera")
window.config(background="#FFFFFF")
imageFrame = tk.Frame(window, width=640, height=640)
imageFrame.grid(row=0, column=0, padx=0, pady=0, columnspan=40, rowspan=50)

# function gui

def setCameraOption():
    try:
        global video_capture
        video_capture = cv2.VideoCapture(0)
        # time.sleep(2)
        show_frame()
    except:
        print(':(s')

def setFileOption():
    try:
        global video_capture
        if e1.get():
            video_capture = cv2.VideoCapture(e1.get())
        # time.sleep(2)
            show_frame()
    except:
        print(':(')

Button(window, text="camera", command=setCameraOption).grid(row=0, column=1)
Button(window, text="File", command=setFileOption).grid(row=0, column=2)
e1 = Entry(window)
e1.grid(row=0, column=3)

# cv2
feelings_faces = []
video_capture = cv2.VideoCapture(0)
# time.sleep(2)

# functions
#  ----
def addEmojiToArray():
  for emotion in Emotions:
    feelings_faces.append(cv2.imread('./emojis/' + emotion.name + '.png', -1))


def readPercentValues():
    # fake
    for emotion in Emotions:
        percentValues[emotion.name] = random.randint(0, 100)
    print(Emotions[max(percentValues, key=lambda key: percentValues[key])])
    return Emotions[max(percentValues, key=lambda key: percentValues[key])].value


# global index for show_frame loop <- useful for make changes every 'X' time
index = IntVar()
readPercentValues()
addEmojiToArray()
index.set(100)

def show_frame():
    global index
    global face_image
    global video_capture
    global isCamera
    ret, frame = video_capture.read()
    selectedEmoji = 1

    if frame is not None:
        cv2image = cv2.resize(frame, (640, 640))
        if (index.get() == 100):
            index.set(0)
            selectedEmoji = readPercentValues()
            # ustawianie emotki
            face_image = feelings_faces[selectedEmoji -1]

        for emotion in Emotions:
            cv2.putText(cv2image, str(emotion.name+ ' ' + str(percentValues[emotion.name]) + '%'), (10, emotion.value * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
            cv2.rectangle(cv2image, (130, emotion.value * 20 + 10), (130 + percentValues[emotion.name], (emotion.value + 1) * 20 + 4), (255, 0, 0), -1)


            for c in range(0, 3):
                  cv2image[200:320, 10:130, c] = face_image[:,:,c] * (face_image[:, :, 3] / 255.0) +  cv2image[200:320, 10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)

        index.set(index.get() + 1)
        cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
        cv2.waitKey(1)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        display1.imgtk = imgtk
        display1.configure(image=imgtk)
    else:
        print('else')
    window.after(30, show_frame)

display1 = tk.Label(imageFrame)
display1.grid(row=1, column=0, padx=10, pady=2)

show_frame()  #Display
window.mainloop()  #Starts GUI
