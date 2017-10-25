import cv2
import random
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

selectedEmotion = 0

class Emotions(Enum):
   smutek = 1
   zaskoczenie = 2
   strach = 3
   podziw = 4
   radosc = 5
   zainteresowanie = 6
   spokoj = 7
   obrzydzenie = 8

feelings_faces = []
video_capture = cv2.VideoCapture(0)

# functions

def addEmojiToArray():
  for emotion in Emotions:
    feelings_faces.append(cv2.imread('./emojis/' + emotion.name + '.png', -1))


def readPercentValues():
    # fake
    for emotion in Emotions:
        percentValues[emotion.name] = random.randint(0, 100)
    return Emotions[max(percentValues, key=lambda key: percentValues[key])].value


def displayFrame():
  readPercentValues()
  addEmojiToArray()
  i = 100

  while True:
    ret, frame = video_capture.read()
    selectedEmoji = 1

    if(i == 100):
        i = 0
        selectedEmoji = readPercentValues()
        print(selectedEmoji)
        # ustawianie emotki
        face_image = feelings_faces[selectedEmoji]

    for emotion in Emotions:
      cv2.putText(frame, str(emotion.name), (10, emotion.value * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
      cv2.rectangle(frame, (130, emotion.value * 20 + 10), (130 + percentValues[emotion.name], (emotion.value + 1) * 20 + 4), (255, 0, 0), -1)


      for c in range(0, 3):
        frame[200:320, 10:130, c] = face_image[:,:,c] * (face_image[:, :, 3] / 255.0) +  frame[200:320, 10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)


    i+=1

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) == 27:
      break
    if cv2.getWindowProperty('Camera', 0) == -1:
      print('x')
      break



# Program
addEmojiToArray()
displayFrame()
video_capture.release()
cv2.destroyAllWindows()