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
 
def lip_distance(shape):
    top_lip=shape[50:53]
    top_lip=np.concatenate((top_lip, shape[61:64]))
    
    low_lip=shape[56:59]
    low_lip=np.concatenate((low_lip, shape[65:68]))
    
    top=np.mean(top_lip, axis=0)
    low=np.mean(low_lip, axis=0)
    
    distance=abs(top[1] - low[1])
    return distance

def main():
    cap = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    face_cascade = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    MOUNTH_THRESH = 25
    MOUNTH_FRAMES = 20
    COUNTER = 0
    
    #### thuc hien vong lap pht hien mat  va mieng ####
    while TRUE:
        ret, frame = cap.read()
        frame=imutils.resize(frame, width=500)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade(gray)
        rects= detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, 
                                         minSize = (30, 30), flags = cv2. CASCADE_SCALE_IMAGE)
    
    ### phat hien khuon mat ####
        for face in faces:
            
            x1=face.left()
            y1=face.top()
            x2=face.right()
            y2=face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
     #### phat hien mieng ###
            
            for (x, y, w, h) in rects:
                rect=dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
                shape = predictor(gray, rect)
                shape=face_utils.shape_to_np(shape)  
                distance= lip_distance(shape)
                print(distance)
                lip=shape[48:60]
                cv2.drawContours(frame,[lip], -1, (0,255,0), 2)
                
                if (distance > MOUNTH_THRESH):
                    cv2.putText(frame,"YAWN ALERT",(10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not alarm_status2 :
                            alarm_status2 =True
                            print('warring warring')
                            t2=threading.Thread(target=sound_alarm, args=('wake up',))
                            t2.daemon=True
                            t2.start()
                else:
                     alarm_status2=False
                     
                     cv2.putText(frame,"YAWN: {:.2f}".format(distance),(300, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                            
        cv2.imshow("Detect Driver State: MOUNT", frame)
        if(cv2.waitKey(1)==ord('q')):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
###### ket thuc ham #####
if __name__ == "__main__":
    main()
      
    
