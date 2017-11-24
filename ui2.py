import random
import sys
from enum import Enum

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QPixmap
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QGraphicsView, QFileDialog
from cv2 import cv2


from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

from PyQt5 import QtCore, QtGui, QtWidgets, QtWebEngineWidgets, QtMultimedia

import glob

import cv2
import numpy as np
import dlib
import os
import glob
import random
import _thread
from multiprocessing import Process

from matplotlib import pyplot as plt

from sklearn import svm
predictor5_path = "./shape_predictor_5_face_landmarks.dat"
predictor68_path = "./shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "./dlib_face_recognition_resnet_model_v1.dat"
faceCascade = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")

detector = dlib.get_frontal_face_detector()
sp5 = dlib.shape_predictor(predictor68_path)
sp68 = dlib.shape_predictor(predictor68_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

percentValues = {
    'neutral': 0,
    'anger': 0,
    'contempt': 0,
    'disgust': 0,
    'fear': 0,
    'happy': 0,
    'sadness': 0,
    'surprise': 0
}

class Emotions(Enum):
   neutral = 0
   anger = 1
   contempt = 2
   disgust = 3
   fear = 4
   happy = 5
   sadness = 6
   surprise = 7

feelings_faces = []

def addEmojiToArray():
  for emotion in Emotions:
    feelings_faces.append(cv2.imread('./emojis/' + emotion.name + '.png', -1))
    print('./emojis/' + emotion.name + '.png')



def readPercentValues():
    # fake
    for emotion in Emotions:
        percentValues[emotion.name] = random.randint(0, 100)
    print(Emotions[max(percentValues, key=lambda key: percentValues[key])])
    return Emotions[max(percentValues, key=lambda key: percentValues[key])]


readPercentValues()
addEmojiToArray()





class Ui_Dialog():
    def __init__(self, Form, url=0, capturing=False):
            self.Form = Form
            self.movie_url = ''
            self.capturing = capturing
            self.url = url
            print("ready")
            # adres serwera
            
    def startCapture(self):
        self.capturing = True
        cap = cv2.VideoCapture(self.url)
        counter = 0
        frameStep = 10
        features = None
        
        while (cap.isOpened()):
            ret, frame = cap.read()
            counter += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frameStep == counter:
                counter = 0
                features = get_features(gray)
            try:
                pred = clf.predict([features])
                print(Emotions(int(str(pred).strip('[').strip(']'))).name)
                self.readPercentValues(Emotions(int(str(pred).strip('[').strip(']'))).name)
            except Exception as err:
                print(err)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        
    def endCapture(self):
        self.capturing = False

    def ready(self):
        print("ready")

    def readPercentValues(self, emotion):       

        if self.radioCamera.isChecked():
            print('camera checked')
        if self.radioMovie.isChecked():
            print('movie checked')
        if self.radioPhoto.isChecked():
            print('photo checked')
        self.smutek_progress.setValue(percentValues['sadness'])
        self.zaskoczenie_progress.setValue(percentValues['surprise'])
        self.strach_progress.setValue(percentValues['fear'])
        self.podziw_progress.setValue(percentValues['contempt'])
        self.radosc_progress.setValue(percentValues['happy'])
        self.anger_progress.setValue(percentValues['anger'])
        self.spokoj_progress.setValue(percentValues['neutral'])
        self.obrzydzenie_progress.setValue(percentValues['disgust'])
        pixmap = QPixmap('emojis/' + emotion + '.png')
        self.emotionIcon.setPixmap(pixmap)

    def startCaptureCamera(self):
        self.url = 0
        try:
            self.startCapture()
        except:
            print('camera dead')

    def stopPlaying(self):
        self.player.pause()
    def play(self):
        self.player.play()
        
    def startVideo(self, url):
        self.url = url
        try:
            self.startCapture()
        except:
            print('camera dead')
    


    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self.Form, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;WMV Files (*.wmv)", options=options)
        if fileName:
            try:
                self.startVideo(fileName)
            except:
                print('something wrong')

    def openFileNameDialogPhoto(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self.Form, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;WMV Files (*.wmv)", options=options)
        if fileName:
            try:
                pixmap = QPixmap(fileName)
                self.photoImage.setPixmap(pixmap)
            except:
                print('something wrong')

    def setupUi(self, Dialog):
            Dialog.setObjectName("Dialog")
            # rozmiar okna
            Dialog.resize(954, 601)

            # tworzenie ramek na przyciski
            self.frame = self.setUpFrame(Dialog, "camera", (10, 10, 241, 461))
            self.frame_2 = self.setUpFrame(Dialog, "movie", (270, 10, 261, 271))
            self.frame_3 = self.setUpFrame(Dialog, "photo", (270, 290, 261, 181))

            # camera buttons
            self.start_button = QtWidgets.QPushButton(self.frame)
            self.start_button.setGeometry(QtCore.QRect(0, 0, 181, 61))
            self.start_button.setObjectName("start")
            self.start_button.clicked.connect(self.startCaptureCamera)

            self.stop = QtWidgets.QPushButton(self.frame)
            self.stop.setGeometry(QtCore.QRect(0, 70, 181, 61))
            self.stop.setObjectName("stop")

            # disconnect camera
            self.stop.clicked.connect(self.endCapture)

            # Photo
            self.photoImage = QtWidgets.QLabel(self.frame_3)
            self.photoImage.setGeometry(QtCore.QRect(0, 60, 700, 500))

            self.get_photo_button = QtWidgets.QPushButton(self.frame_3)
            self.get_photo_button.setGeometry(QtCore.QRect(120, 0, 60, 61))
            self.get_photo_button.setObjectName("start")
            self.get_photo_button.clicked.connect(self.openFileNameDialogPhoto)

            #video
            # movies buttons
            self.start_movie_button = QtWidgets.QPushButton(self.frame_2)
            self.start_movie_button.setGeometry(QtCore.QRect(500, 0, 181, 61))
            self.start_movie_button.setObjectName("start")
            self.start_movie_button.clicked.connect(self.openFileNameDialog)

            self.stop_movie_button = QtWidgets.QPushButton(self.frame_2)
            self.stop_movie_button.setGeometry(QtCore.QRect(500, 70, 181, 61))
            self.stop_movie_button.setObjectName("stop")
            self.stop_movie_button.clicked.connect(self.stopPlaying)

            self.play_movie_button = QtWidgets.QPushButton(self.frame_2)
            self.play_movie_button.setGeometry(QtCore.QRect(500, 140, 181, 61))
            self.play_movie_button.setObjectName("play")
            self.play_movie_button.clicked.connect(self.play)

            self.player = QMediaPlayer(self.frame_2)
            self.video = QVideoWidget(self.frame_2)
            self.graphicsView = QGraphicsView(self.frame_2)
            self.graphicsView.scene()
            self.graphicsView.setStyleSheet("background-color: #000;")
            self.graphicsView.setViewport(self.video)
            self.graphicsView.setGeometry(QtCore.QRect(0, 0, 500, 500))
            self.graphicsView.show()
            self.video.setStyleSheet("background-color: #000;")
            # self.player.setMedia(QMediaContent(QUrl('test.wmv')))
            self.player.setVideoOutput(self.video)
            self.player.setPosition(200)
            self.video.setGeometry(QtCore.QRect(100, 100, 1000, 1000))
            self.video.show()

            self.tabWidget = QtWidgets.QTabWidget(Dialog)
            self.tabWidget.setEnabled(True)
            self.tabWidget.setGeometry(QtCore.QRect(0, 0, 701, 561))
            self.tabWidget.setObjectName("tabWidget")

            self.tabWidget.addTab(self.frame, "")
            self.tabWidget.addTab(self.frame_2, "")
            self.tabWidget.addTab(self.frame_3, "")

            self.smutek_progress = QtWidgets.QProgressBar(Dialog)
            self.smutek_progress.setGeometry(QtCore.QRect(820, 50, 118, 23))
            self.smutek_progress.setProperty("value", 24)
            self.smutek_progress.setObjectName("smutek_progress")
            self.smutek_label = QtWidgets.QLabel(Dialog)
            self.smutek_label.setGeometry(QtCore.QRect(730, 60, 47, 13))
            self.smutek_label.setObjectName("smutek_label")
            self.zaskoczenie_label = QtWidgets.QLabel(Dialog)
            self.zaskoczenie_label.setGeometry(QtCore.QRect(730, 100, 71, 16))
            self.zaskoczenie_label.setObjectName("zaskoczenie_label")
            self.zaskoczenie_progress = QtWidgets.QProgressBar(Dialog)
            self.zaskoczenie_progress.setGeometry(QtCore.QRect(820, 90, 118, 23))
            self.zaskoczenie_progress.setProperty("value", 24)
            self.zaskoczenie_progress.setObjectName("zaskoczenie_progress")
            self.strach_label = QtWidgets.QLabel(Dialog)
            self.strach_label.setGeometry(QtCore.QRect(730, 140, 47, 13))
            self.strach_label.setObjectName("strach_label")
            self.strach_progress = QtWidgets.QProgressBar(Dialog)
            self.strach_progress.setGeometry(QtCore.QRect(820, 130, 118, 23))
            self.strach_progress.setProperty("value", 24)
            self.strach_progress.setObjectName("strach_progress")
            self.podziw_label = QtWidgets.QLabel(Dialog)
            self.podziw_label.setGeometry(QtCore.QRect(730, 180, 47, 13))
            self.podziw_label.setObjectName("podziw_label")
            self.podziw_progress = QtWidgets.QProgressBar(Dialog)
            self.podziw_progress.setGeometry(QtCore.QRect(820, 170, 118, 23))
            self.podziw_progress.setProperty("value", 24)
            self.podziw_progress.setObjectName("podziw_progress")
            self.radosc_label = QtWidgets.QLabel(Dialog)
            self.radosc_label.setGeometry(QtCore.QRect(730, 220, 47, 13))
            self.radosc_label.setObjectName("radosc_label")
            self.radosc_progress = QtWidgets.QProgressBar(Dialog)
            self.radosc_progress.setGeometry(QtCore.QRect(820, 210, 118, 23))
            self.radosc_progress.setProperty("value", 24)
            self.radosc_progress.setObjectName("radosc_progress")
            self.anger_label = QtWidgets.QLabel(Dialog)
            self.anger_label.setGeometry(QtCore.QRect(730, 251, 91, 20))
            self.anger_label.setObjectName("anger_label")
            self.anger_progress = QtWidgets.QProgressBar(Dialog)
            self.anger_progress.setGeometry(QtCore.QRect(820, 250, 118, 23))
            self.anger_progress.setProperty("value", 24)
            self.anger_progress.setObjectName("anger_progress")
            self.spokoj_progress = QtWidgets.QProgressBar(Dialog)
            self.spokoj_progress.setGeometry(QtCore.QRect(820, 289, 118, 23))
            self.spokoj_progress.setProperty("value", 24)
            self.spokoj_progress.setObjectName("spokoj_progress")
            self.spokoj_label = QtWidgets.QLabel(Dialog)
            self.spokoj_label.setGeometry(QtCore.QRect(730, 290, 91, 20))
            self.spokoj_label.setObjectName("spokoj_label")
            self.obrzydzenie_progress = QtWidgets.QProgressBar(Dialog)
            self.obrzydzenie_progress.setGeometry(QtCore.QRect(820, 329, 118, 23))
            self.obrzydzenie_progress.setProperty("value", 24)
            self.obrzydzenie_progress.setObjectName("obrzydzenie_progress")
            self.obrzydzenie_label = QtWidgets.QLabel(Dialog)
            self.obrzydzenie_label.setGeometry(QtCore.QRect(730, 330, 91, 20))
            self.obrzydzenie_label.setObjectName("obrzydzenie_label")

            self.readEmotions = QtWidgets.QPushButton(Dialog)
            self.readEmotions.setGeometry(QtCore.QRect(700, 390, 110, 50))
            self.readEmotions.clicked.connect(self.readPercentValues)

            self.stopReadEmotions = QtWidgets.QPushButton(Dialog)
            self.stopReadEmotions.setGeometry(QtCore.QRect(700, 460, 110, 50))
            self.stopReadEmotions.clicked.connect(self.endCapture)

            self.radioCamera = QtWidgets.QRadioButton(Dialog)
            self.radioCamera.setGeometry(QtCore.QRect(700, 520, 41, 17))
            self.radioCamera.setObjectName("radioCamera")
            self.radioCameraLabel = QtWidgets.QLabel(Dialog)
            self.radioCameraLabel.setGeometry(QtCore.QRect(720, 510, 101, 31))
            self.radioCameraLabel.setObjectName("radioCameraLabel")
            self.radioCamera.setChecked(True)


            self.radioMovie = QtWidgets.QRadioButton(Dialog)
            self.radioMovie.setGeometry(QtCore.QRect(700, 540, 41, 17))
            self.radioMovie.setObjectName("radioMovie")
            self.radioMovieLabel = QtWidgets.QLabel(Dialog)
            self.radioMovieLabel.setGeometry(QtCore.QRect(720, 530, 101, 31))
            self.radioMovieLabel.setObjectName("radioCameraLabel")

            self.radioPhoto = QtWidgets.QRadioButton(Dialog)
            self.radioPhoto.setGeometry(QtCore.QRect(700, 560, 41, 17))
            self.radioPhoto.setObjectName("radioPhoto")
            self.radioPhotoLabel = QtWidgets.QLabel(Dialog)
            self.radioPhotoLabel.setGeometry(QtCore.QRect(720, 550, 101, 31))
            self.radioPhotoLabel.setObjectName("radioCameraLabel")


            self.emotionIcon = QtWidgets.QLabel(Dialog)
            pixmap = QPixmap('emojis/spokoj.png')
            self.emotionIcon.setPixmap(pixmap)
            self.emotionIcon.setGeometry(QtCore.QRect(820, 390, 120, 120))


            self.retranslateUi(Dialog)
            QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
            _translate = QtCore.QCoreApplication.translate
            # nadanie przyciskom tekst
            Dialog.setWindowTitle(_translate("Dialog", "StreetView"))
            self.tabWidget.setTabText(self.tabWidget.indexOf(self.frame), _translate("Dialog", "Camera"))
            self.tabWidget.setTabText(self.tabWidget.indexOf(self.frame_2), _translate("Dialog", "Movie"))
            self.tabWidget.setTabText(self.tabWidget.indexOf(self.frame_3), _translate("Dialog", "Photo"))
            self.smutek_label.setText(_translate("Dialog", "smutek"))
            self.zaskoczenie_label.setText(_translate("Dialog", "zaskoczenie"))
            self.strach_label.setText(_translate("Dialog", "strach"))
            self.podziw_label.setText(_translate("Dialog", "podziw"))
            self.radosc_label.setText(_translate("Dialog", "radosc"))
            self.anger_label.setText(_translate("Dialog", "anger"))
            self.spokoj_label.setText(_translate("Dialog", "spokoj"))
            self.obrzydzenie_label.setText(_translate("Dialog", "obrzydzenie"))
            self.radioCameraLabel.setText(_translate("Dialog", "camera"))
            self.radioMovieLabel.setText(_translate("Dialog", "movie"))
            self.radioPhotoLabel.setText(_translate("Dialog", "photo"))
            self.start_button.setText(_translate("Dialog", "Start camera"))
            self.get_photo_button.setText(_translate("Dialog", "get photo"))
            self.play_movie_button.setText(_translate("Dialog", "play"))
            self.stop.setText(_translate("Dialog", "Stop camera"))
            self.start_movie_button.setText(_translate("Dialog", "Get movie"))
            self.readEmotions.setText(_translate("Dialog", "Start read emotions"))
            self.stopReadEmotions.setText(_translate("Dialog", "Stop read emotions"))
            self.stop_movie_button.setText(_translate("Dialog", "Stop movie"))


    def setUpFrame(self, Dialog, name, geometry):
        frame = QtWidgets.QFrame(Dialog)
        frame.setGeometry(QtCore.QRect(geometry[0], geometry[1], geometry[2], geometry[3]))
        frame.setAutoFillBackground(False)
        frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        frame.setFrameShadow(QtWidgets.QFrame.Raised)
        frame.setObjectName(name)
        return frame

            
def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in Emotions:
        training, prediction = get_files(emotion.name)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            current_feats = get_features(gray)
            if current_feats:
                training_data.append(current_feats) #append image feats to training data list
                training_labels.append(emotion.value)
    
        for item in prediction: #repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            current_feats = get_features(gray)
            if current_feats:
                prediction_data.append(current_feats)
                prediction_labels.append(emotion.value)

    return training_data, training_labels, prediction_data, prediction_labels
            
def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset\\%s\\*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction
    
def run_recognizer():
    print("uczenie!!!")
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
#     training_data, training_labels, prediction_data, prediction_labels = sets_made  # to speed up TEMP
    
    print ("training SVM classifier")
    print ("size of training set is:", len(training_labels), "images")
    print ("size of test set is:", len(prediction_labels), "images")
    
    X = training_data
    y = training_labels

    clf = svm.LinearSVC(C=5)
    # clf = svm.SVC(C=0.8, kernel='rbf', gamma=20)  # it's worse
    clf.fit(X, y)
    print ("predicting classification set")
    cnt = 0
    correct = 0
    incorrect = 0
    
    for image in training_data:
        pred = clf.predict([image]) #predict emotion
        if pred == training_labels[cnt]: #validate it
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1
    print("Training accuracy:", ((100*correct)/(correct + incorrect))) 

    cnt = 0
    correct = 0
    incorrect = 0
    for image in prediction_data:
        pred = clf.predict([image]) #predict emotion
        if pred == prediction_labels[cnt]: #validate it
            correct += 1
            cnt += 1
        else:
            incorrect += 1
            cnt += 1

    # return ((100 * correct) / (correct + incorrect))
    return clf
    
def get_features(im):
    faces = faceCascade.detectMultiScale(im)
    if len(faces) == 0:
        return None  # no face detected :(

    for (x, y, w, h) in faces:
        cur_feat = []
        x_l = x
        x_r = x + w
        y_t = y
        y_b = y + h
        d = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        width = x_r - x_l
        height = y_b - y_t
        shape = sp68(im, d)
        for i in range(0, 68):
            feat_x = (shape.part(i).x - x_l) / width
            feat_y = (shape.part(i).y - y_t) / height
            cur_feat.append(feat_x)
            cur_feat.append(feat_y)
    return cur_feat

if __name__ == "__main__":
    clf = run_recognizer()
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    Form = QtWidgets.QWidget()
    ui = Ui_Dialog(Form)
    # Form.show()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())