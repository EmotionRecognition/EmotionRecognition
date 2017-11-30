from sklearn import svm
import dlib
from cv2 import cv2 
import glob
import random
from enum import Enum

class Emotions(Enum):
   neutral = 0
   anger = 1
   contempt = 2
   disgust = 3
   fear = 4
   happy = 5
   sadness = 6
   surprise = 7

predictor5_path = "./shape_predictor_5_face_landmarks.dat"
predictor68_path = "./shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "./dlib_face_recognition_resnet_model_v1.dat"
faceCascade = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_default.xml")

detector = dlib.get_frontal_face_detector()
sp5 = dlib.shape_predictor(predictor68_path)
sp68 = dlib.shape_predictor(predictor68_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)



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
if __name__ == '__main__':
    clf = run_recognizer()