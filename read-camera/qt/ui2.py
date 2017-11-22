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

class Capture():
    def __init__(self):
        self.capturing = False
        self.url = 0

    def setUrl(self, url):
        self.url = url
        print(url)

    def startCapture(self):
        self.capturing = True
        cap = cv2.VideoCapture(self.url)
        while(self.capturing):
            ret, frame = cap.read()
            cv2.imshow("Capture", frame)
            cv2.waitKey(5)
        cv2.destroyAllWindows()

    def endCapture(self):
        self.capturing = False


class Ui_Dialog():
        def __init__(self, Form):
            self.Form = Form
            self.movie_url = ''
            print("ready")
            # adres serwera

        def ready(self):
            print("ready")

        def readPercentValues(self):
            imageName = readPercentValues()

            if self.radioCamera.isChecked():
                print('camera checked')
            if self.radioMovie.isChecked():
                print('movie checked')
            if self.radioPhoto.isChecked():
                print('photo checked')
            self.smutek_progress.setValue(percentValues['smutek'])
            self.zaskoczenie_progress.setValue(percentValues['zaskoczenie'])
            self.strach_progress.setValue(percentValues['strach'])
            self.podziw_progress.setValue(percentValues['podziw'])
            self.radosc_progress.setValue(percentValues['radosc'])
            self.zainteresowanie_progress.setValue(percentValues['zainteresowanie'])
            self.spokoj_progress.setValue(percentValues['spokoj'])
            self.obrzydzenie_progress.setValue(percentValues['obrzydzenie'])
            pixmap = QPixmap('emojis/' + imageName.name + '.png')
            self.emotionIcon.setPixmap(pixmap)

        def startCaptureCamera(self):
            self.capture.url = 0
            try:
                self.capture.startCapture()
            except:
                print('camera dead')

        def stopPlaying(self):
            self.player.pause()
        def play(self):
            self.player.play()


        def openFileNameDialog(self):
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            fileName, _ = QFileDialog.getOpenFileName(self.Form, "QFileDialog.getOpenFileName()", "",
                                                      "All Files (*);;WMV Files (*.wmv)", options=options)
            if fileName:
                try:
                    self.player.setMedia(QMediaContent(QUrl(fileName)))
                    self.player.play()
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
                self.capture = Capture()
                self.start_button = QtWidgets.QPushButton(self.frame)
                self.start_button.setGeometry(QtCore.QRect(0, 0, 181, 61))
                self.start_button.setObjectName("start")
                self.start_button.clicked.connect(self.startCaptureCamera)

                self.stop = QtWidgets.QPushButton(self.frame)
                self.stop.setGeometry(QtCore.QRect(0, 70, 181, 61))
                self.stop.setObjectName("stop")

                # disconnect camera
                self.stop.clicked.connect(self.capture.endCapture)

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
                self.zainteresowanie_label = QtWidgets.QLabel(Dialog)
                self.zainteresowanie_label.setGeometry(QtCore.QRect(730, 251, 91, 20))
                self.zainteresowanie_label.setObjectName("zainteresowanie_label")
                self.zainteresowanie_progress = QtWidgets.QProgressBar(Dialog)
                self.zainteresowanie_progress.setGeometry(QtCore.QRect(820, 250, 118, 23))
                self.zainteresowanie_progress.setProperty("value", 24)
                self.zainteresowanie_progress.setObjectName("zainteresowanie_progress")
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
                self.stopReadEmotions.clicked.connect(self.capture.endCapture)

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
                self.zainteresowanie_label.setText(_translate("Dialog", "zainteresowanie"))
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


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    Form = QtWidgets.QWidget()
    ui = Ui_Dialog(Form)
    # Form.show()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())