import numpy as np
import warnings
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import os
import pathlib
import tensorflow as tf
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import argparse 

warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


################################################################################################
# Argument Parsing

parser = argparse.ArgumentParser()
parser.add_argument('--json', default=False, const=True, nargs='?')
parser.add_argument('--weights', default=False, const=True, nargs='?') 
args = parser.parse_args()
json_path = args.json
weights = args.weights

################################################################################################
with open(json_path, 'r') as json_file:
    json_savedModel= json_file.read()

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = tf.keras.models.model_from_json(json_savedModel)
classifier.load_weights(weights)
class_labels = ["Angry", "Disgusted", "Worried", "Happy", "Sad", "Terrified"]
cap = cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi=roi_gray.astype('float')/255.0
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)
            preds=classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position=(x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
cap.release()
cv2.destroyAllWindows()