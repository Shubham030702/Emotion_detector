import cv2 as cv
import numpy as np
import mediapipe as mp

mface = mp.solutions.face_detection
face = mface.FaceDetection()
mdraw = mp.solutions.drawing_utils

capture = cv.VideoCapture(0)

capture.set(3,1400)
capture.set(4,1100)

from tensorflow.keras.models import load_model

model = load_model("Models\Model_best_3.h5")

classes = ['angry', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

while True:
    istrue,frame = capture.read()
    imgrgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    result = face.process(imgrgb)
    if result.detections:
        for detect in result.detections:
            bboxc=detect.location_data.relative_bounding_box
            ih,iw,ix = frame.shape
            bbox = int(bboxc.xmin*iw),int(bboxc.ymin*ih),\
                int(bboxc.width*iw),int(bboxc.height*ih)
            roi = frame[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
            input_size = (48, 48) 
            roi_resized = cv.resize(roi, input_size)
            roi_transformed = np.expand_dims(roi_resized, axis=0)
            roi_normalized = roi_transformed / 255.0
            pred = model.predict(roi_normalized)
            prediction = np.argmax(pred)
            cv.rectangle(frame,bbox,(255,0,255),2)
    cv.putText(frame,f"Emotion:-{classes[prediction]}",(bbox[0],bbox[1]-10),cv.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
    cv.imshow("webcam",frame)
    if cv.waitKey(20) & 0xff == ord('d'):
        break

capture.release()
cv.destroyAllWindows()