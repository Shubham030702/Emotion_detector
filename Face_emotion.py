import cv2 as cv
import numpy as np
import time
import os 
import Face_mesh_module as fm

capture = cv.VideoCapture(0)
capture.set(3,1400)
capture.set(4,1100)

detector = fm.Mesh_detector()
while True:
    istrue,frame = capture.read()
    detector.findFaceMesh(frame)
    cv.imshow("WebCam",frame)
    if cv.waitKey(20) & 0xff == ord('d'):
        break

capture.release()
cv.destroyAllWindows()