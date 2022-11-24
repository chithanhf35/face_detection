
from pickle import TRUE
from sys import flags
import cv2
import numpy as np
import dlib
import imutils
import time
import os
import signal
from threading import Thread
from pyparsing import lineStart
from scipy.spatial import distance as dist
from imutils import face_utils
import threading
import playsound
from playsound import playsound

##### ham am thanh ####
def sound_alarm(path):
	# play an alarm sound
	playsound(r'C:\Users\chith\OneDrive\Desktop\2022_Python\alert.mp3')
    

#### KHAI BAO HAM ####
def calc_EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear= (A + B) / (2.0 * C)
    return ear
def lip_distance(shape):
    top_lip=shape[50:53]
    top_lip=np.concatenate((top_lip, shape[61:64]))
    
    low_lip=shape[56:59]
    low_lip=np.concatenate((low_lip, shape[65:68]))
    
    top=np.mean(top_lip, axis=0)
    low=np.mean(low_lip, axis=0)
    
    distance=abs(top[1] - low[1])
    return distance

def results_EAR(shape):
    
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEye =shape[lStart:lEnd]
    rightEye=shape[rStart:rEnd]
    
    leftEAR =calc_EAR(leftEye)
    rightEAR=calc_EAR(rightEye)
    
    ear= (leftEAR + rightEAR) / 2.0 
    return(ear, leftEye, rightEye)
    
def main():
    cap = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    face_cascade = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    ##### dat nguonng mat #####
    EYE_THRESH = 0.25
    EYE_FRAMES = 20
    COUNTER = 0
    MOUNTH_THRESH = 25
    
    #### thuc hien vong lap pht hien mat ####
    while TRUE:
        ret, frame = cap.read()
        frame=imutils.resize(frame, width=500)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade(gray)
        rects= detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                         minSize = (30, 30), flags = cv2. CASCADE_SCALE_IMAGE)
        ### phat hien mat ####
        for face in faces:
            
            x1=face.left()
            y1=face.top()
            x2=face.right()
            y2=face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        #rects = detector(gray, 0)
    
            #for rect in rects:
            for (x, y, w, h) in rects:
                
                rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
                
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                eye = results_EAR(shape)
                ear = eye[0]
                print(ear)
                leftEye = eye [1]
                rightEye = eye[2]

                distance = lip_distance(shape)
                print(distance)

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                lip = shape[48:60]
                cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

                if ear <= EYE_THRESH:
                    print(EYE_THRESH)
                    COUNTER += 1
                    print(COUNTER)

                    if COUNTER >= EYE_FRAMES:
                        if alarm_status == False:
                            alarm_status = True
                            t = Thread(target=sound_alarm, args=('wake up sir',))
                            t.daemon = True
                            t.start()

                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                else:
                    COUNTER = 0
                    alarm_status = False

                if (distance > MOUNTH_THRESH):
                        cv2.putText(frame, "Yawn Alert", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if alarm_status2 == False:
                            alarm_status2 = True
                            t = Thread(target=sound_alarm, args=('take some fresh air sir',))
                            t.daemon = True
                            t.start()
                else:
                    alarm_status2 = False

                # Add the text on the frame
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Detect Driver State", frame)
        if(cv2.waitKey(1)==ord('q')):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
###### ket thuc ham #####
if __name__ == "__main__":
    main()
