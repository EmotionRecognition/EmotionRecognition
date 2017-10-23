import cv2
import numpy as np

# Detekcja twarzy na zdjęciu termo

if __name__ == "__main__":

    face_cascade = cv2.CascadeClassifier("haar/haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye.xml')

    img = cv2.imread("emo.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            print (ex,ey,ew,eh)
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.imshow('gray', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()