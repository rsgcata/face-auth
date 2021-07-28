import cv2
import numpy as np
from PIL import Image
import os
from pprint import pprint

path = 'faceImages'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# function to get the images and label data
def trainUserData(userId):
    faceSamples=[]
    ids = []

    idPath = 'faceImages' + os.path.sep + str(userId)    
    imagePaths = [os.path.join(idPath,fl) for fl in os.listdir(idPath)]

    for imagePath in imagePaths:
        # Open image in grayscale mode (L)
        pillowImage = Image.open(imagePath)
        
        numpyImage = np.array(pillowImage,'uint8')
        faceSamples.append(numpyImage)
        ids.append(int(userId))
        '''
        faces = detector.detectMultiScale(numpyImage)
        for (x,y,w,h) in faces:
            faceSamples.append(numpyImage[y:y+h,x:x+w])
            ids.append(int(userId))
        '''

    recognizer.train(faceSamples, np.array(ids))
    recognizer.write('trainedData' + os.path.sep + str(userId) + '.trainer.yml')