import cv2
import numpy as numpy
from PIL import image
import os

path='dataset'

recognizer=cv2.face.LBPHFaceRecognizer_create()
detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLables(path):

    imagePaths=[os.path.join(path,f) for in os.listdir(path)]
    facesamples=[]
    ids=[]

    for imagepath in imagePaths:
        PIL_img=Image.open(imagepath).convert('L')
        img_numpy=np.array(PIL_img, 'uint8')

        id=int(os.path.split(imagepath[-1]).split(".")[1])
        faces=detector.detectMultiScale(img_numpy)

        for(x,y,w,h) in faces:
            facesamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return facesamples.ids

print("\n[INFO] trainig faces.....")
faces.ids=getImagesAndLables(path)
recognizer.train(faces, np.array(ids))

recognizer.write('Trainer/ trainer.yml')

print("\n [INFO] {0} faces trained.".format(len(np.unique(ids))))


