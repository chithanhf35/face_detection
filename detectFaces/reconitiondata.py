from cProfile import Profile
from sys import getprofile
import cv2 
import numpy as np
import os.path
import sqlite3
from PIL import Image

# Get profile by ID
face_cascade= cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read(r'C:\Users\chith\OneDrive\Desktop\2022_Python\Recognizer\trainningData.yml')

def getProfile(id):
    conn=sqlite3.connect('data.db')
    query="SELECT * FROM sampling WHERE ID=" + str(id)
    cusror= conn.execute(query)
    
    profile=None
    for row in cusror:
        profile= row
    conn.close()
    return profile

# detect face
cap=cv2.VideoCapture(0)
fontFace=cv2.FONT_HERSHEY_SIMPLEX

while(True):
    
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #faces=face_cascade.dectectMutilScale(gray)
    faces=face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
    
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        roi_gray=gray[y:y+h, x:x+w]
        id, confidence= recognizer.predict(roi_gray)
        
        if confidence < 40:
            profile=getProfile(id)
            
            if(profile != None):
                cv2.putText(frame, ""+str(profile[1]), (x+10,h+y+30), fontFace,1, (0,255,0),2 )
            else:
                cv2.putText(frame,"Unknown",fontFace,1,(0,255,0),3)
                
    cv2.imshow('imge',frame)
    if(cv2.waitKey(1)==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()




            
