import dlib
import cv2
import numpy as np


faceCascade = cv2.CascadeClassifier("C:\\Users\Mateusz Gutowski\Desktop\emotionRecoDlib\haarcascades\haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("C:\\Users\Mateusz Gutowski\Desktop\emotionRecoDlib\shape_predictor_68_face_landmarks.dat")

image = cv2.imread("face1.jpg")

cap = cv2.VideoCapture(0)
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.05,
                                         minNeighbors=5,
                                         minSize=(100, 100),
                                         flags=cv2.CASCADE_SCALE_IMAGE
                                         )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Converting the OpenCV rectangle coordinates to Dlib rectangle
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        landmarks = np.matrix([[p.x, p.y]
                               for p in predictor(frame, dlib_rect).parts()])

    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])

        cv2.putText(frame, str(idx), pos,

                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))

        cv2.circle(frame, pos, 2, color=(0, 255, 255), thickness=-1)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

