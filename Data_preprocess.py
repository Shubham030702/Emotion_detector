import cv2 as cv
import numpy as np
import os
import pandas as pd
import Face_mesh_module as fm

DIR = "D:/Opencv/Advance Opencv topics/EMOTION_DETECTOR/images"

detector = fm.Mesh_detector()

def rescaleFrame(frame, size=0.25):
    width = int(frame.shape[1] * size)  # frame.shape[1] is the width of the frame we have passed
    height = int(frame.shape[0] * size) # frame.shape[0] is the height of the frame we have passed
    dimension = (width, height)
    return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)

# Create a list to store the data
data = []

# Iterate through the images and extract landmarks
for i in os.listdir(DIR):
    path = os.path.join(DIR, i)
    for img in os.listdir(path):
        npath = os.path.join(path, img)
        image = cv.imread(npath)
        if image is not None:
            image = rescaleFrame(image)
            frame, landms = detector.findFaceMesh(image)
            
            if landms:
                # Prepare the data row
                row = [img] + [coord for landmark in landms[0] for coord in landmark]
                data.append(row)
                
                print(f"Processed: {img}, Landmarks count: {len(landms[0])}")
                
        if cv.waitKey(20) & 0xff == ord("d"):
            break
    if cv.waitKey(20) & 0xff == ord("d"):
        break

# Create a DataFrame
columns = ['image_name'] + [f'landmark_{i}_{axis}' for i in range(468) for axis in ('x', 'y')]
df = pd.DataFrame(data, columns=columns)

# Save the DataFrame to a CSV file
df.to_csv('landmarks.csv', index=False)
