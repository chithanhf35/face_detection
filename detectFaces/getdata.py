import cv2 
import numpy as np
import sqlite3
import os.path
import time

#truy cap du dieu SQL
def insertOrupdate(id, name):
    conn=sqlite3.connect(r'C:\Users\chith\OneDrive\Desktop\2022_Python\data.db')
    query="SELECT * FROM Sampling WHERE ID =" + str(id)
    cusror=conn.execute(query)
    isRecordExist=0
    for row in cusror :
        isRecordExist=1

    if(isRecordExist==0):
        query="INSERT INTO Sampling (ID,Name) VALUES ("+str(id) + ",'"+ str(name)+"')"
    else:
        query="UPDATE Sampling SET Name='"+str(name)+"'WHERE ID="+str(id)
    conn.execute(query)
    conn.commit()
    conn.close()

def capture_demo():
    #load tv
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

    #insert to db (database)
    id=input("enter your id: ")
    name=input("enter your name: ")
    insertOrupdate(id,name)
    samplenum=0
    cap=cv2.VideoCapture(0)
    while(True):
            time.sleep(0.3)
            ret,frame=cap.read()
            gray=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces= face_cascade.detectMultiScale(gray, 1.3,5)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,225,0),2)
                print()
                if not os.path.exists('dataSet'):
                    os.makedirs('dataSet')
                samplenum +=1
                cv2.imwrite('dataSet/user.' + str(id) + '.' + str(samplenum) + '.jpg',gray[y: y+h, x:x+w])
            cv2.imshow('frame',frame)
            cv2.waitKey(1)
            if samplenum > 50:
                break
    cap.release()
    cv2.destroyAllWindows()
capture_demo()

