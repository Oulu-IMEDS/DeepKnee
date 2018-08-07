from xray_processor import process_file
import os
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='../../DICOM_TEST/imgs/')
    parser.add_argument("--save_dir", default='../../DICOM_TEST/rois/')
    parser.add_argument("--detections", default='../../detection_results.txt')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    detections = np.loadtxt(args.detections, dtype=str)
    
    for i in range(detections.shape[0]):
        fname, bbox = detections[i][0], detections[i][1:].astype(int)
        # read KL grades for this file somewhere
        # ATTENTION: these grades are FAKE and you need to retrieve them depending on the filename
        process_file(i, fname, args.data_dir, args.save_dir, bbox, 5, 5)
        print(fname, bbox)
    
