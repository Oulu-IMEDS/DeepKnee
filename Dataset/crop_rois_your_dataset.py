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

    detections = np.loadtxt(args.detections, dtype=str, ndmin=2)
    
    for i in range(detections.shape[0]):
        fname, bbox = detections[i][0], detections[i][1:].astype(np.int)

        # ATTENTION: the KL grades specified below are FAKE and you need to
        #            retrieve the correct values by yourself (e.g. extract
        #            from the filenames or pass from metadata dataframe)

        process_file(i, fname, args.data_dir, args.save_dir, bbox, 5, 5)
        print(fname, bbox)
