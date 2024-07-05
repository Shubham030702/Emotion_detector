import cv2 as cv
import numpy as np
import os
import Face_mesh_module as fm

DIR = "D:\Opencv\Advance Opencv topics\FaceEmotion_detector\images"

detector = fm.Mesh_detector()

def rescaleFrame(frame , size = 0.25):
    width = int(frame.shape[1]*size)  # frame.shape[1] is the width of frame we have passed
    height = int(frame.shape[0]*size) # frame.shape[0] is the height of frame we have passed
    dimension = (width,height)
    return cv.resize(frame,dimension,interpolation=cv.INTER_AREA)

for i in os.listdir(DIR):
    path = os.path.join(DIR,i);
    for img in os.listdir(os.path.join(path)):
        npath = os.path.join(path,img)
        image = cv.imread(npath)
        image=rescaleFrame(image)
        frame,landms = detector.findFaceMesh(image)
        print(landms[0],img)
        if cv.waitKey(20) & 0xff == ord("d"):
            break
    if cv.waitKey(20) & 0xff == ord("d"):
            break

print(dict)
    