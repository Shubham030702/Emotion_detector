#  this is the module for face mesh

import cv2 as cv
import mediapipe as mp
import time

class Mesh_detector:
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.mpmesh = mp.solutions.face_mesh
        self.mesh = self.mpmesh.FaceMesh(
                    static_image_mode=self.staticMode,
                    max_num_faces=self.maxFaces,
                    min_detection_confidence=self.minDetectionCon,
                    min_tracking_confidence=self.minTrackCon
                )       
        self.mpdraw = mp.solutions.drawing_utils
        self.drawspecs = self.mpdraw.DrawingSpec(thickness=1,circle_radius=1)
    
    def findFaceMesh(self,frame,draw = True):
        img_rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        self.result = self.mesh.process(img_rgb)
        faces =  []
        if self.result.multi_face_landmarks:
            for landms in self.result.multi_face_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(frame,landms,self.mpmesh.FACEMESH_CONTOURS,self.drawspecs,self.drawspecs)
                face = []
                for id,lm in enumerate(landms.landmark):
                    ih,iw,ic = frame.shape
                    x,y = int(lm.x*iw),int(lm.y*ih)
                    face.append([x,y])
        faces.append(face)
        return frame,faces


def main():
    capture = cv.VideoCapture(0)

    ptime = 0
    detector = Mesh_detector()
    while True:
        istrue,frame = capture.read()
        if not istrue:
            break
        frame,detect = detector.findFaceMesh(frame)
        if len(detect)!=0:
            print(detect[0])
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv.putText(frame,f"Fps = {str(int(fps))}",(30,70),cv.FONT_HERSHEY_COMPLEX,1,(255,0,255),3)
        cv.imshow("WebCam",frame)
        if cv.waitKey(20) & 0xff==ord('d'):
            break
    
    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()