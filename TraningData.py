import cv2 
import numpy as np
import os
from PIL import Image
import sqlite3

recognizer= cv2.face.LBPHFaceRecognizer_create()
path='dataSet'

def getImgdWithID(path):
    
    imagPaths = [os.path.join(path,f) for f in os.listdir(path)]
    print(imagPaths)
    faces=[]
    IDs=[]
    for imagPath in imagPaths:
        faceImg=Image.open(imagPath).convert('L')
        faceNp=np.array(faceImg,'uint8')
        print(faceNp)
        Id=int(imagPath.split('\\')[1].split('.')[1])
        faces.append(faceNp)
        IDs.append(Id)
        cv2.imshow('trainning',faceNp)
        cv2.waitKey(20)
    return faces,IDs
    
faces,IDs=getImgdWithID(path)
#train data
recognizer.train(faces,np.array(IDs))
#train data
recognizer.train(faces,np.array(IDs))
if not os.path.exists('Recognizer'):
    os.makedirs('Recognizer')
recognizer.save('Recognizer/trainningData.yml')

cv2.destroyAllWindows()

