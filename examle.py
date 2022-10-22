import cv2 
import numpy as np
import os
from PIL import Image
import sqlite3

recognizer= cv2.face.LBPHFaceRecognizer_create()
path='Dataset'

def getImgdWithID(path):
    imagPaths = [os.path.join(path,f) for f in os.listdir(path)]
    print(imagPaths)
    faces=[]
    IDs=[]
    for imagPath in imagPaths:
        faceImg=Image.open(imagPath).convert('L')
        faceNp=np.array(faceImg,'uint8')
        print(faceNp)
getImgdWithID(path)