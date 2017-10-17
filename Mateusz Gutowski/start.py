import numpy as np
import cv2
#q  - wyjscie z aplikacji


class EmotionRecognition():

    def getVideo(self):
        cap = cv2.VideoCapture(0)
        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Display the resulting frame
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()





if (__name__ == "__main__"):

    #er = EmotionRecognition()
    #er.getVideo
    locFace = "C:\\Users\Mateusz Gutowski\Desktop\emotion recognition\haarcascades\haarcascade_frontalface_default.xml"
    locEye = "C:\\Users\Mateusz Gutowski\Desktop\emotion recognition\haarcascades\haarcascade_eye.xml"
    locSmile = "C:\\Users\Mateusz Gutowski\Desktop\emotion recognition\haarcascades\haarcascade_smile.xml"
    face_cascade = cv2.CascadeClassifier(locFace)
    eye_cascade = cv2.CascadeClassifier(locEye)
    smile_cascade = cv2.CascadeClassifier(locSmile)

    img = cv2.imread('face0.jpg')

    # cv2.imshow('gray',gray)
    # cv2.waitKey(0)

    cap = cv2.VideoCapture(0)
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color =  frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            smiles = smile_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
              cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            for (ex, ey, ew, eh) in smiles:
              cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



