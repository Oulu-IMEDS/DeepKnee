"""
Inference script for a custom data

(c) Aleksei Tiulpin, University of Oulu, 2017
"""

import os
import argparse
import glob
from collections import OrderedDict

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

from ouludeepknee.train.model import KneeNet
from ouludeepknee.train.augmentation import CenterCrop
from ouludeepknee.train.dataset import get_pair


def load_model(filename, net:torch.nn.Module):
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


def load_img(fname, img_proc, patch_proc):
    img = Image.open(fname)
    # We will use 8bit
    tmp = np.array(img, dtype=float)
    img = Image.fromarray(np.uint8(255 * (tmp / 65535.)))

    img = img_proc(img)

    l, m = get_pair(img)

    lateral_patch = patch_proc(l)
    medial_patch = patch_proc(m)

    return lateral_patch, medial_patch


SNAPSHOTS_KNEE_GRADING = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../snapshots_knee_grading'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',  default='../../DICOM_TEST/rois/5/')
    parser.add_argument('--save_results', default='../../KL_grading_results.txt')
    parser.add_argument('--snapshots',  default=SNAPSHOTS_KNEE_GRADING)
    parser.add_argument('--bw', type=int, default=64)

    args = parser.parse_args()

    print('Version of pytorch:', torch.__version__)

    mean_vector, std_vector = np.load(os.path.join(args.snapshots, 'mean_std.npy'))
    snapshots_fnames = glob.glob(os.path.join(args.snapshots, '*', '*.pth'))

    models = []
    for snp_name in snapshots_fnames:
        tmp = load_model(snp_name, KneeNet(args.bw, 0.2, True))
        tmp.eval()
        models.append(tmp)

    normTransform = transforms.Normalize(mean_vector, std_vector)

    patch_transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x.float(),
        normTransform,
    ])

    imgs_fnames = glob.glob(os.path.join(args.dataset, '**', '*.png'))
    sm = torch.nn.Softmax(1)
    preds = []
    for fname in tqdm(imgs_fnames, total=len(imgs_fnames)):
        inp = load_img(fname, CenterCrop(300), patch_transform)

        lateral = inp[0].unsqueeze(0)
        medial = inp[1].unsqueeze(0)
        res = 0
        for m in models:
            res += m(lateral, medial)
        res = sm(res).data.squeeze().numpy()

        preds.append([fname.split('/')[-1], ] + res.tolist())

    with open(args.save_results, 'w') as f:
        for pred in preds:
            f.write(f'{pred[0]} {pred[1]:.5f} {pred[2]:.5f} {pred[3]:.5f} {pred[4]:.5f} {pred[5]:.5f}')
            f.write('\n')
