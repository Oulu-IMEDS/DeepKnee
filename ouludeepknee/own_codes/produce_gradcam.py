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
from torch.autograd import Variable
import torchvision.transforms as transforms
from tqdm import tqdm

from ouludeepknee.own_codes.model import KneeNet
from ouludeepknee.own_codes.dataset import get_pair
from ouludeepknee.own_codes.augmentation import CenterCrop


if torch.cuda.is_available():
    maybe_cuda = 'cuda'
else:
    maybe_cuda = 'cpu'


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
    def __init__(self, snapshots_paths, mean_std_path):
        super().__init__()
        self.states = []
        for snap_path in snapshots_paths:
            self.states.append(torch.load(snap_path, map_location=maybe_cuda))

        self.net1 = None
        self.net2 = None
        self.net3 = None

        self.grads_l1 = None
        self.grads_m1 = None

        self.grads_l2 = None
        self.grads_m2 = None

        self.grads_l3 = None
        self.grads_m3 = None
        self.sm = torch.nn.Softmax(1)
        self.mean_std_path = mean_std_path

        if torch.cuda.is_available():
            self.cuda()

    def init_networks_from_states(self):
        mean_vector, std_vector = np.load(self.mean_std_path)
        normTransform = transforms.Normalize(mean_vector, std_vector)
        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            lambda x: x.float(),
            normTransform,
        ])

        nets = []
        for state in self.states:
            if torch.cuda.is_available():
                net = nn.DataParallel(KneeNet(64, 0.2, True)).cuda()
            else:
                net = nn.DataParallel(KneeNet(64, 0.2, True))
            net.load_state_dict(state)
            nets.append(net.module)

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

    def load_picture(self, fname, nbits=16, flip_left=False):
        """

        :param fname: str or numpy.ndarray
            Takes either full path to the image or the numpy array
        :return:
        """

        if isinstance(fname, str):
            img = Image.open(fname)
        elif isinstance(fname, np.ndarray):
            img = fname
            if nbits == 16:
                img = Image.fromarray(np.uint8(255 * (img / 65535.)))
            elif nbits == 8:
                if img.dtype != np.uint8:
                    raise TypeError
                img = Image.fromarray(img)
            else:
                raise TypeError
        else:
            raise TypeError

        width, height = img.size

        if width != 350 or height != 350:
            img = img.resize((350, 350), Image.BICUBIC)

        if flip_left:
            if '_L' in fname.split('/')[0]:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        cropper = CenterCrop(300)

        l, m = get_pair(cropper(img))

        l = self.patch_transform(l)
        m = self.patch_transform(m)

        return cropper(img), l.view(1, 1, 128, 128), m.view(1, 1, 128, 128)

    @staticmethod
    def decompose_forward_avg(net, l, m):
        l_o = net.branch(l)
        m_o = net.branch(m)

        concat = torch.cat([l_o, m_o], 1)
        o = net.final(concat.view(l.size(0), net.final.in_features))
        return l_o, m_o, o

    @staticmethod
    def extract_features_branch(net, l, m, wl, wm):
        def weigh_maps(weights, maps):
            maps = Variable(maps.squeeze())
            weights = weights.squeeze()

            if torch.cuda.is_available():
                res = torch.zeros(maps.size()[-2:]).cuda()
            else:
                res = Variable(torch.zeros(maps.size()[-2:]))

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
        l_o1, m_o1, o1 = self.decompose_forward_avg(self.net1, l, m)
        l_o1.register_hook(lambda grad: self.grads_l1.append(grad))
        m_o1.register_hook(lambda grad: self.grads_m1.append(grad))
        # Net 2
        l_o2, m_o2, o2 = self.decompose_forward_avg(self.net2, l, m)
        l_o2.register_hook(lambda grad: self.grads_l2.append(grad))
        m_o2.register_hook(lambda grad: self.grads_m2.append(grad))
        # Net 3
        l_o3, m_o3, o3 = self.decompose_forward_avg(self.net3, l, m)
        l_o3.register_hook(lambda grad: self.grads_l3.append(grad))
        m_o3.register_hook(lambda grad: self.grads_m3.append(grad))

        return o1 + o2 + o3

    def predict(self, x, nbits=16, flip_left=False):
        """Makes a prediction from file or a pre-loaded image

        :param x: str or numpy.array
        :param nbits: int
            By default we load 16 bit images produced by CropROI Object and convert them to 8bit.
        :return: tuple
            Image, Heatmap, probabilities
        """
        self.init_networks_from_states()
        img, l, m = self.load_picture(x, nbits=nbits, flip_left=flip_left)
        self.train(True)
        self.zero_grad()

        if torch.cuda.is_available():
            out = self.forward(Variable(l.cuda()), Variable(m.cuda()))
        else:
            out = self.forward(Variable(l), Variable(m))

        probs = self.sm(out).data.cpu().numpy()

        ohe = OneHotEncoder(sparse=False, n_values=5)
        index = np.argmax(out.cpu().data.numpy(), axis=1).reshape(-1, 1)

        if torch.cuda.is_available():
            out.backward(torch.from_numpy(ohe.fit_transform(index)).float().cuda())
        else:
            out.backward(torch.from_numpy(ohe.fit_transform(index)).float())

        if torch.cuda.is_available():
            heatmap = self.compute_gradcam(
                Variable(l.cuda()), Variable(m.cuda()), 300, 128, 7)
        else:
            heatmap = self.compute_gradcam(
                Variable(l), Variable(m), 300, 128, 7)

        return img, heatmap, probs.squeeze()

    def predict_save(self, fileobj_in, nbits=16, fname_suffix=None, path_dir_out='./',
                     flip_left=False):
        if fname_suffix is not None:
            pass
        elif isinstance(fileobj_in, str):
            fname_suffix = os.path.splitext(os.path.basename(fileobj_in))[0]
        else:
            fname_suffix = ''

        img, heatmap, probs = self.predict(x=fileobj_in, nbits=nbits, flip_left=flip_left)
        if flip_left:
            img = np.fliplr(img)
            heatmap = np.fliplr(heatmap)

        plt.figure(figsize=(7, 7))
        plt.imshow(np.asarray(img), cmap=plt.cm.Greys_r)
        plt.imshow(heatmap, cmap=plt.cm.jet, alpha=0.3)
        plt.xticks([])
        plt.yticks([])
        tmp_fname = os.path.join(path_dir_out, f'heatmap_{fname_suffix}.png')
        plt.savefig(tmp_fname, bbox_inches='tight', dpi=300, pad_inches=0)
        plt.close()

        plt.figure(figsize=(7, 1))
        for kl in range(5):
            plt.text(kl - 0.2, 0.35, "%.2f" % np.round(probs[kl], 2), fontsize=15)
        plt.bar(np.array([0, 1, 2, 3, 4]), probs, color='red', align='center',
                tick_label=['KL0', 'KL1', 'KL2', 'KL3', 'KL4'], alpha=0.3)
        plt.ylim(0, 1)
        plt.yticks([])
        tmp_fname = os.path.join(path_dir_out, f'prob_{fname_suffix}.png')
        plt.savefig(tmp_fname, bbox_inches='tight', dpi=300, pad_inches=0)
        plt.close()
        return probs.squeeze().argmax()


SNAPSHOTS_KNEE_GRADING = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../snapshots_knee_grading'))

SNAPSHOTS_EXPS = ['2017_10_10_12_30_42', '2017_10_10_12_30_46', '2017_10_10_12_30_49']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_folds', default=SNAPSHOTS_KNEE_GRADING)
    parser.add_argument('--path_input')
    parser.add_argument('--nbits', type=int, default=16)
    parser.add_argument('--flip_left', type=bool, default=False)
    parser.add_argument('--path_output', default='../')
    parser.add_argument('--path_output_csv', default='../preds.csv')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config = parse_args()

    avg_preds = {}
    labels = {}
    nets_snapshots_names = []

    for snp in SNAPSHOTS_EXPS:
        nets_snapshots_names.extend(glob(os.path.join(config.path_folds, snp, '*.pth')))

    net = KneeNetEnsemble(nets_snapshots_names,
                          mean_std_path=os.path.join(SNAPSHOTS_KNEE_GRADING, 'mean_std.npy'))

    paths_test_files = glob(os.path.join(config.path_input, '*', '*.png'))

    os.makedirs(config.path_output, exist_ok=True)

    with open(config.path_output_csv, 'w') as f:
        f.write('IMG,predicted\n')
        for path_test_file in tqdm(paths_test_files, total=len(paths_test_files)):
            pred = net.predict_save(fileobj_in=path_test_file, nbits=config.nbits,
                                    path_dir_out=config.path_output, flip_left=config.flip_left)
            line = '{},{}\n'.format(path_test_file.split('/')[-1], pred)
            f.write(line)


