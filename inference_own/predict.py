"""
Inference script for a custom dataset


(c) Aleksei Tiulpin, University of Oulu, 2017
"""

import sys
sys.path.insert(0, '../own_codes/')
import os
import argparse
import numpy as np
import glob
import torch
from collections import OrderedDict
from model import KneeNet
from dataset import get_pair


def load_model(filename, net):
    state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
    try:
        net.load_state_dict(state_dict)
    except:

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    return net


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
        tmp = load_model(snp_name, KneeNet(args.bw, 0.2, True))
        tmp.eval()
        models.append(tmp)
