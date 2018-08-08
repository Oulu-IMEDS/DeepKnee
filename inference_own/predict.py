"""
Inference script for a custom dataset


(c) Aleksei Tiulpin, University of Oulu, 2017
"""

import sys
import os
import argparse
import numpy as np
import glob
import torch

sys.path.insert(0, '../own_codes/')
from model import KneeNet
from dataset import get_pair

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',  default='../../DICOM_TEST/rois')
    parser.add_argument('--snapshots',  default='../snapshots_knee_grading')
    parser.add_argument('--bw', type=int, default=64)
    args = parser.parse_args()

    print('Version of pytorch:', torch.__version__)

    mean_std = np.load(os.path.join(args.snapshots, 'mean_std.npy'))
    snapshots_fnames = glob.glob(os.path.join(args.snapshots, '*', '*.pth'))

    models = []
    for snp_name in snapshots_fnames:
        tmp = KneeNet(args.bw, 0.2, True)
        weights = torch.load(snp_name, map_location=lambda storage, loc: storage)
        tmp.load_state_dict(weights)
        tmp.eval()
        models.append(tmp)
        