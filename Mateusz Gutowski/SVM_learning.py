
import cv2
import numpy as np
import dlib
import os
import glob
import random
import _thread
from multiprocessing import Process


from matplotlib import pyplot as plt

predictor5_path = "./shape_predictor_5_face_landmarks.dat"
predictor68_path = "./shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "./dlib_face_recognition_resnet_model_v1.dat"
faceCascade = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")

detector = dlib.get_frontal_face_detector()
sp5 = dlib.shape_predictor(predictor68_path)
sp68 = dlib.shape_predictor(predictor68_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
def get_features(im):
    # dets = detector(im, 1)
    faces = faceCascade.detectMultiScale(im)
    # print("Faces:",faces)
    if len(faces) == 0:
        return None  # no face detected :(

#     for d in dets:
#         # TODO: what if there are two faces in one pic --> seems ok
# #         print(d)
#         cur_feat = []
#         x_l = d.left()
#         x_r = d.right()
#         y_t = d.top()
#         y_b = d.bottom()
    for (x, y, w, h) in faces:
        # TODO: what if there are two faces in one pic --> seems ok
        #         print(d)
        cur_feat = []
        x_l = x
        x_r = x + w
        y_t = y
        y_b = y + h
        d = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        width = x_r - x_l
        height = y_b - y_t
    #     cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 0), im.shape[0]//150)
        shape = sp68(im, d)
        for i in range(0, 68):
            # TODO: normalize to 0 mean? How?
            feat_x = (shape.part(i).x - x_l) / width
            feat_y = (shape.part(i).y - y_t) / height
            cur_feat.append(feat_x)
            cur_feat.append(feat_y)

    #         cv2.circle(orig, (shape.part(i).x, shape.part(i).y), 1, (255, 0, 0), im.shape[0]//150)
    #     feats.append(cur_feat)
    return cur_feat

from sklearn import svm

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset\\%s\\*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            current_feats = get_features(gray)
            if current_feats:
                training_data.append(current_feats) #append image feats to training data list
                training_labels.append(emotions.index(emotion))
    
        for item in prediction: #repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            current_feats = get_features(gray)
            if current_feats:
                prediction_data.append(current_feats)
                prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


# sets_made = make_sets()

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


    return clf




# image = cv2.imread("face1.jpg")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
# image = get_features(gray)
# pred = clf.predict([image])
# print(pred)


check_new_emotion = True
gray = None

def video_go():
    global gray
    global check_new_emotion
    frame_step = 0
    counter = 0
    cap = cv2.VideoCapture(0)
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if counter == frame_step:
            check_new_emotion = True
            counter = 0
        counter += 1

        features = get_features(gray)
        try:
            pred = clf.predict([features])
            print(pred)
        except:
            None


        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def features_check():
    global check_new_emotion
    global clf
    pred = [0]
    while True:
        if check_new_emotion:
            features = get_features(gray)
            # check_new_emotion = False


            try:
                pred = clf.predict([features])
                print(pred)
            except:
                None
        print(pred)


# if __name__ == '__main__':
#     mp.set_start_method('spawn')
#     q = mp.Queue()
#     p = mp.Process(target=foo, args=(q,))
#     p.start()
#     print(q.get())
#     p.join()



#_thread        TNIE!
clf = run_recognizer()

# metascore = []
# for i in range(0, 3):
#     correct = run_recognizer()
#     print("Test: got", correct, "percent correct!")
#     metascore.append(correct)
#
while 1:
    video_go()
    # features_check()


# try:
#    _thread.start_new_thread(features_check, ())
#    _thread.start_new_thread(video_go,())
#
# except:
#    print ("Psykro mi - nie dziala")
#
# while 1:
#    pass
