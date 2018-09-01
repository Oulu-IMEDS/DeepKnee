import os
import argparse
from copy import deepcopy
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from model import KneeNet

from dataset import get_pair
from augmentation import CenterCrop


if torch.cuda.is_available():
    maybe_cuda = 'cuda'
else:
    maybe_cuda = 'cpu'


def load_picture16bit(fname):
    patch_transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x.float(),
        normTransform,
    ])

    img = Image.open(fname)

    tmp = np.array(img, dtype=float)
    img = Image.fromarray(np.uint8(255 * (tmp / 65535.)))

    cropper = CenterCrop(300)

    l, m = get_pair(cropper(img))

    l = patch_transform(l)
    m = patch_transform(m)

    return cropper(img), l.view(1, 1, 128, 128), m.view(1, 1, 128, 128)


def smooth_edge_mask(s, w):
    res = np.zeros((s + w * 2, s + w * 2))
    res[w:w + s, w:w + s] = 1
    res = cv2.blur(res, (2 * w, 2 * w))

    return res[w:w + s, w:w + s]


def inverse_pair_mapping(l, m, s, ps=128, smoothing=7):
    pad = int(np.floor(s / 3))

    l = cv2.resize(l, (ps, ps), cv2.INTER_CUBIC)
    l *= smooth_edge_mask(l.shape[0], smoothing)

    m = cv2.resize(m, (ps, ps), cv2.INTER_CUBIC)
    m *= smooth_edge_mask(m.shape[0], smoothing)

    hm = np.zeros((s, s))
    hm[pad:pad + ps, 0:ps] = l
    hm[pad:pad + ps, s - ps:] = m

    return hm


class KneeNetEnsemble(nn.Module):
    def __init__(self, nets):
        super().__init__()
        net1 = nets[0]
        net1.final = nets[0].final[1]

        net2 = nets[1]
        net2.final = nets[1].final[1]

        net3 = nets[2]
        net3.final = nets[2].final[1]

        self.net1 = deepcopy(net1)
        self.net2 = deepcopy(net2)
        self.net3 = deepcopy(net3)

        self.grads_l1 = []
        self.grads_m1 = []

        self.grads_l2 = []
        self.grads_m2 = []

        self.grads_l3 = []
        self.grads_m3 = []

    def decopmpose_forward_avg(self, net, l, m):
        l_o = net.branch(l)
        m_o = net.branch(m)

        concat = torch.cat([l_o, m_o], 1)
        o = net.final(concat.view(l.size(0), 512))
        return l_o, m_o, o

    def extract_features_branch(self, net, l, m, wl, wm):
        def weigh_maps(weights, maps):
            maps = maps.squeeze()
            weights = weights.squeeze()

            res = torch.zeros(maps.size()[-2:]).to(maybe_cuda)

            for i, w in enumerate(weights):
                res += w * maps[i]

            return res

        # We need to re-assemble the architecture
        branch = nn.Sequential(net.branch.block1,
                               nn.MaxPool2d(2),
                               net.branch.block2,
                               nn.MaxPool2d(2),
                               net.branch.block3)

        o_l = branch(l).data
        o_m = branch(m).data
        # After extracting the features, we weigh them based on the provided weights
        o_l = weigh_maps(wl, o_l)
        o_m = weigh_maps(wm, o_m)
        return F.relu(o_l), F.relu(o_m)

    def compute_gradcam(self, l, m, img_size, ps, smoothing=7):
        wl, wm = self.grads_l1[0].data, self.grads_m1[0].data
        ol1, om1 = self.extract_features_branch(self.net1, l, m, wl, wm)

        wl, wm = self.grads_l2[0].data, self.grads_m2[0].data
        ol2, om2 = self.extract_features_branch(self.net1, l, m, wl, wm)

        wl, wm = self.grads_l3[0].data, self.grads_m3[0].data
        ol3, om3 = self.extract_features_branch(self.net1, l, m, wl, wm)

        l_out = (ol1 + ol2 + ol3) / 3.

        m_out = (om1 + om2 + om3) / 3.

        heatmap = inverse_pair_mapping(l_out.data.cpu().numpy(),
                                       np.fliplr(m_out.data.cpu().numpy()),
                                       img_size, ps, smoothing)

        heatmap -= heatmap.min()
        heatmap /= heatmap.max()

        return heatmap

    def forward(self, l, m):
        self.grads_l1 = []
        self.grads_m1 = []

        self.grads_l2 = []
        self.grads_m2 = []

        self.grads_l3 = []
        self.grads_m3 = []

        # Producing the branch outputs and registering the corresponding hooks for attention maps
        # Net 1
        l_o1, m_o1, o1 = self.decopmpose_forward_avg(self.net1, l, m)
        l_o1.register_hook(lambda grad: self.grads_l1.append(grad))
        m_o1.register_hook(lambda grad: self.grads_m1.append(grad))
        # Net 2
        l_o2, m_o2, o2 = self.decopmpose_forward_avg(self.net2, l, m)
        l_o2.register_hook(lambda grad: self.grads_l2.append(grad))
        m_o2.register_hook(lambda grad: self.grads_m2.append(grad))
        # Net 3
        l_o3, m_o3, o3 = self.decopmpose_forward_avg(self.net3, l, m)
        l_o3.register_hook(lambda grad: self.grads_l3.append(grad))
        m_o3.register_hook(lambda grad: self.grads_m3.append(grad))

        return o1 + o2 + o3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_folds', default='../snapshots_knee_grading/')
    parser.add_argument('--path_input')
    parser.add_argument('--path_output', default='../')

    args = parser.parse_args()
    args.snapshots = ['2017_10_10_12_30_42', '2017_10_10_12_30_46', '2017_10_10_12_30_49']

    return args


if __name__ == '__main__':
    config = parse_args()

    print('WARNING: Implemented and tested on pytorch==0.4.0 only')

    mean_vector, std_vector = np.load('../snapshots_knee_grading/mean_std.npy')
    normTransform = transforms.Normalize(mean_vector, std_vector)
    patch_transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x.float(),
        normTransform,
    ])

    avg_preds = {}
    labels = {}
    nets = []

    for fold in config.snapshots:
        for snap_path in glob(os.path.join(config.path_folds, fold, '*.pth')):
            net = nn.DataParallel(KneeNet(64, 0.2, True)).to(maybe_cuda)
            net.load_state_dict(torch.load(snap_path, map_location=maybe_cuda))
            nets.append(deepcopy(net.module))

    net = nn.DataParallel(KneeNetEnsemble(nets))
    net.to(maybe_cuda)

    # Producing the GradCAM output using the equations provided in the article
    paths_test_files = glob(os.path.join(config.path_input, '**', '*.png'))

    if not os.path.exists(config.path_output):
        os.makedirs(config.path_output)

    for path_test_file in paths_test_files:
        img, l, m = load_picture16bit(path_test_file)

        net.train(True)
        net.zero_grad()
        out = net.module(l.to(maybe_cuda), m.to(maybe_cuda))
        ohe = OneHotEncoder(sparse=False, n_values=5)
        index = np.argmax(out.cpu().data.numpy(), axis=1).reshape(-1, 1)
        out.backward(torch.from_numpy(ohe.fit_transform(index)).float().to(maybe_cuda))

        heatmap = net.module.compute_gradcam(
            l.to(maybe_cuda), m.to(maybe_cuda), 300, 128, 7)

        plt.figure(figsize=(7, 7))
        plt.imshow(np.asarray(img), cmap=plt.cm.Greys_r)
        plt.imshow(heatmap, cmap=plt.cm.jet, alpha=0.3)
        plt.xticks([])
        plt.yticks([])
        tmp_fname = os.path.join(config.path_output,
                                 'heatmap_' + os.path.basename(path_test_file))
        plt.savefig(tmp_fname, bbox_inches='tight', dpi=300, pad_inches=0)
        plt.close()

        plt.figure(figsize=(7, 1))
        probs = F.softmax(out).cpu().data[0].numpy()
        for kl in range(5):
            plt.text(kl - 0.2, 0.35, "%.2f" % np.round(probs[kl], 2), fontsize=15)
        plt.bar(np.array([0, 1, 2, 3, 4]), probs, color='red', align='center',
                tick_label=['KL0', 'KL1', 'KL2', 'KL3', 'KL4'], alpha=0.3)
        plt.ylim(0, 1)
        plt.yticks([])
        tmp_fname = os.path.join(config.path_output,
                                 'prob_' + os.path.basename(path_test_file))
        plt.savefig(tmp_fname, bbox_inches='tight', dpi=300, pad_inches=0)
        plt.close()
