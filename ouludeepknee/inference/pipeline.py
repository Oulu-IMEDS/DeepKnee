import argparse
import glob
import io
import os
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from ouludeepknee.inference.utils import fuse_bn_recursively
from ouludeepknee.train.augmentation import CenterCrop
from ouludeepknee.train.dataset import get_pair
from ouludeepknee.train.model import KneeNet


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
    def __init__(self, snapshots_paths, mean_std_path, device=None):
        super().__init__()
        self.states = []
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.device = device
        for snap_path in snapshots_paths:
            self.states.append(torch.load(snap_path, map_location=self.device))

        self.cropper = CenterCrop(300)
        self.ohe = OneHotEncoder(sparse=False, categories=[range(5)])

        mean_vector, std_vector = np.load(mean_std_path)
        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            lambda x: x.float(),
            transforms.Normalize(mean_vector, std_vector)
        ])

        self.grads_l1 = None
        self.grads_m1 = None

        self.grads_l2 = None
        self.grads_m2 = None

        self.grads_l3 = None
        self.grads_m3 = None
        self.sm = torch.nn.Softmax(1)
        self.mean_std_path = mean_std_path

        self.init_networks_from_states()

    def empty_gradient_arrays(self):
        # Initializing arrays for storing the gradients
        self.grads_l1, self.grads_m1 = [], []
        self.grads_l2, self.grads_m2 = [], []
        self.grads_l3, self.grads_m3 = [], []

    def init_networks_from_states(self):
        models = {}
        for idx, state in enumerate(self.states):
            # Data Parallel was accidentally stored back in 2017.
            model = nn.DataParallel(KneeNet(64, 0.2, False)).to(self.device)
            model.load_state_dict(state)
            # Converting data parallel into a regular model
            model = model.module
            # Removing the dropout
            model.final = model.final[1]
            # Fusing BatchNorm
            # We need to re-assemble the architecture so that we are able to extract features
            # We should not forget that after calling model.branch, we should also pass the result to self.avgpool
            branch = nn.Sequential(model.branch.block1,
                                   nn.MaxPool2d(2),
                                   model.branch.block2,
                                   nn.MaxPool2d(2),
                                   model.branch.block3)
            branch = fuse_bn_recursively(branch)
            model.branch = branch
            models[f'net{idx + 1}'] = deepcopy(model)

        self.__dict__['_modules'].update(models)
        self.to(self.device)

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

        img_cropped = self.cropper(img)
        lateral, medial = get_pair(img_cropped)

        lateral = self.patch_transform(lateral).to(self.device)
        medial = self.patch_transform(medial).to(self.device)

        return img_cropped, lateral.view(1, 1, 128, 128), medial.view(1, 1, 128, 128)

    @staticmethod
    def decompose_forward_avg(net, l, m):
        # Reducing the memory footprint.
        # We don't really need gradients to compute the features
        with torch.no_grad():
            l_o = net.branch(l)
            m_o = net.branch(m)
            l_o_avg = F.adaptive_avg_pool2d(l_o, (1, 1))
            m_o_avg = F.adaptive_avg_pool2d(m_o, (1, 1))
        # These variables will requre features as they will initiate the forward pass to the FC layer
        # From which we will get the gradients
        l_o_avg.requires_grad = True
        m_o_avg.requires_grad = True
        # A normal forward pass. Concatenating the outputs from the lateral and the medial sides
        concat = torch.cat([l_o_avg, m_o_avg], 1)
        # Passing the results through an FC layer
        o = net.final(concat.view(l.size(0), net.final.in_features))
        return l_o, m_o, l_o_avg, m_o_avg, o

    def weigh_maps(self, weights, maps):
        maps = maps.squeeze()
        weights = weights.squeeze()

        res = torch.zeros(maps.size()[-2:]).to(self.device)

        for i, w in enumerate(weights):
            res += w * maps[i]
        return res

    def extract_gradcam_weighted_maps(self, o_l, o_m, wl, wm):
        # After extracting the features, we weigh them based on the provided weights
        o_l = self.weigh_maps(wl, o_l)
        o_m = self.weigh_maps(wm, o_m)
        return F.relu(o_l), F.relu(o_m)

    def compute_gradcam(self, features, img_size, ps, smoothing=7):
        w_lateral, w_medial = self.grads_l1[0].data, self.grads_m1[0].data
        ol1, om1 = self.extract_gradcam_weighted_maps(features['net1'][0], features['net1'][1], w_lateral, w_medial)

        w_lateral, w_medial = self.grads_l2[0].data, self.grads_m2[0].data
        ol2, om2 = self.extract_gradcam_weighted_maps(features['net2'][0], features['net2'][1], w_lateral, w_medial)

        w_lateral, w_medial = self.grads_l3[0].data, self.grads_m3[0].data
        ol3, om3 = self.extract_gradcam_weighted_maps(features['net3'][0], features['net3'][1], w_lateral, w_medial)

        l_out = (ol1 + ol2 + ol3) / 3.
        m_out = (om1 + om2 + om3) / 3.

        heatmap = inverse_pair_mapping(l_out.detach().to('cpu').numpy(),
                                       np.fliplr(m_out.detach().to('cpu').numpy()),
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
        l_o1, m_o1, l_o1_avg, m_o1_avg, o1 = self.decompose_forward_avg(self.net1, l, m)
        l_o1_avg.register_hook(lambda grad: self.grads_l1.append(grad))
        m_o1_avg.register_hook(lambda grad: self.grads_m1.append(grad))
        # Net 2
        l_o2, m_o2, l_o2_avg, m_o2_avg, o2 = self.decompose_forward_avg(self.net2, l, m)
        l_o2_avg.register_hook(lambda grad: self.grads_l2.append(grad))
        m_o2_avg.register_hook(lambda grad: self.grads_m2.append(grad))
        # Net 3
        l_o3, m_o3, l_o3_avg, m_o3_avg, o3 = self.decompose_forward_avg(self.net3, l, m)
        l_o3_avg.register_hook(lambda grad: self.grads_l3.append(grad))
        m_o3_avg.register_hook(lambda grad: self.grads_m3.append(grad))

        features = {'net1': (l_o1, m_o1), 'net2': (l_o2, m_o2), 'net3': (l_o3, m_o3)}

        return o1 + o2 + o3, features

    def predict(self, x, nbits=16, flip_left=False):
        """Makes a prediction from file or a pre-loaded image

        :param x: str or numpy.array
            Image. Should be 130x130mm with the pixel spacing of 0.3mm (300x300 pixels).
        :param nbits: int
            By default we load 16 bit images produced by CropROI Object and convert them to 8bit.
        :param flip_left: bool
            Whether to flip image. Done for the left knees
        :return: tuple
            Image, Heatmap, probabilities
        """
        img, l, m = self.load_picture(x, nbits=nbits, flip_left=flip_left)
        self.empty_gradient_arrays()
        self.train(True)
        self.zero_grad()

        out, features = self.forward(l, m)

        probs = self.sm(out).to('cpu').detach().numpy()
        index = np.argmax(out.detach().to('cpu').numpy(), axis=1).reshape(-1, 1)
        out.backward(torch.from_numpy(self.ohe.fit_transform(index)).float().to(self.device))
        gradcam_heatmap = self.compute_gradcam(features, 300, 128, 7)

        return img, gradcam_heatmap, probs.squeeze()

    def predict_draw(self, fileobj_in, nbits=16, fname_suffix=None, path_dir_out=None, flip_left=False):
        """Makes a prediction from file or a pre-loaded image

        :param fileobj_in: str or numpy.array
            Image. Should be 130x130mm with the pixel spacing of 0.3mm (300x300 pixels).
        :param nbits: int
            By default we load 16 bit images produced by CropROI Object and convert them to 8bit.
        :param fname_suffix: str or None
            Base filename used to save the results
        :param path_dir_out: str or None
            Where to save the heatmap and the softmax barplot
        :param flip_left: bool
            Whether to flip image. Done for the left knees
        :return: tuple
            Image, Heatmap, probabilities
        """
        if fname_suffix is not None:
            pass
        elif isinstance(fileobj_in, str):
            fname_suffix = os.path.splitext(os.path.basename(fileobj_in))[0]
        else:
            fname_suffix = ''

        img, heatmap, probs = self.predict(x=fileobj_in, nbits=nbits, flip_left=flip_left)
        img = np.asarray(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if flip_left:
            img = np.fliplr(img)
            heatmap = np.fliplr(heatmap)
        # overlay with original image
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        img_overlayed = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
        # If path provided, we save the heatmap. Otherwise, this is skipped
        if path_dir_out is not None:
            tmp_fname = os.path.join(path_dir_out, f'heatmap_{fname_suffix}.png')
            cv2.imwrite(tmp_fname, img_overlayed)
        img_overlayed = cv2.cvtColor(img_overlayed, cv2.COLOR_BGR2RGB)

        # Making a bar plot for displaying probabilities
        plt.figure(figsize=(6, 1))
        for kl in range(5):
            plt.text(kl - 0.2, 0.35, "%.2f" % np.round(probs[kl], 2), fontsize=15)
        plt.bar(np.array([0, 1, 2, 3, 4]), probs, color='red', align='center',
                tick_label=['KL0', 'KL1', 'KL2', 'KL3', 'KL4'], alpha=0.3)
        plt.ylim(0, 1)
        plt.yticks([])
        # Saving the figure to a BytesIO object.
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', dpi=100, pad_inches=0)
        buf.seek(0)
        probs_bar_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        plt.close()
        # Now decoding the result from the bytes object
        probs_bar = cv2.imdecode(probs_bar_arr, 1)
        if path_dir_out is not None:
            tmp_fname = os.path.join(path_dir_out, f'prob_{fname_suffix}.png')
            cv2.imwrite(tmp_fname, probs_bar)
        probs_bar = cv2.cvtColor(probs_bar, cv2.COLOR_BGR2RGB)

        return img, img_overlayed, probs_bar, probs.squeeze().argmax()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshots_path', default='../../snapshots_knee_grading')
    parser.add_argument('--images', type=str, default='')
    parser.add_argument('--write_heatmaps', type=bool, default=False)
    parser.add_argument('--nbits', type=int, default=16)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--flip_left', type=bool, default=False)
    parser.add_argument('--output_dir', default='../../../deepknee_test_output', help='Stores heatmaps')
    parser.add_argument('--output_csv', default='../../../deepknee_test_output/preds.csv', help='Stores predictions')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print('Version of pytorch:', torch.__version__)

    args = parse_args()

    avg_preds = {}
    labels = {}

    nets_snapshots_names = glob.glob(os.path.join(args.snapshots_path, "*", '*.pth'))

    net = KneeNetEnsemble(nets_snapshots_names,
                          mean_std_path=os.path.join(args.snapshots_path, 'mean_std.npy'),
                          device=args.device)

    paths_test_files = glob.glob(os.path.join(args.images, '*.png'))

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.output_csv, 'w') as f:
        f.write('IMG,predicted\n')
        for path_test_file in tqdm(paths_test_files, total=len(paths_test_files)):
            image, image_heatmap, preds_bar, pred = net.predict_draw(fileobj_in=path_test_file, nbits=args.nbits,
                                                                     path_dir_out=args.output_dir if args.write_heatmaps else None,
                                                                     flip_left=args.flip_left)

            line = '{},{}\n'.format(path_test_file.split('/')[-1], pred)
            f.write(line)
