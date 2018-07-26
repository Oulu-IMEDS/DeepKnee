from xray_processor import process_file
import os
import numpy as np

if __name__ == "__main__":
    DS_DIR = '/mnt/nas/MOST/Images/XR'
    TO_SAVE = 'KL_dataset/'

    detections = np.loadtxt('MOST_train.csv', dtype=str)
    # Use this file to determine which knee was badly detected
    # number indicates the line numer in "the array detections"
    bad_detections = set(np.loadtxt('MOST_train_poor_detections.csv',dtype=str).tolist())

    for i in range(2):
        fname, bbox = detections[i][0], detections[i][1:].astype(int)
        # read KL grades for this file somewhere
        # ATTENTION: these grades are FAKE and you need to retrieve them depending on the filename
        gradeL, gradeR = 5, 5
        process_file(i, fname, DS_DIR, TO_SAVE, bbox, gradeL, gradeR)
        print(fname, bbox)
    
