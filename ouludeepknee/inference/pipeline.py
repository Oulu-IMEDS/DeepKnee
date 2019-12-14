import argparse
import glob
import io
import os
from copy import deepcopy

import matplotlib

matplotlib.use('Agg')

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
import requests
from pydicom import dcmread
from pydicom.filebase import DicomBytesIO
import logging
import base64

from ouludeepknee.data.utils import read_dicom, process_xray

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
        self.logger = logging.getLogger(f'deepknee-backend:pipeline')
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.device = device
        for snap_path in snapshots_paths:
            self.states.append(torch.load(snap_path, map_location=self.device))
            self.logger.log(logging.INFO, f'Loaded weights from {snap_path}')

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
        self.logger.log(logging.INFO, f'Gradient arrays have been emptied')

    def init_networks_from_states(self):
        models = {}
        for idx, state in enumerate(self.states):
            # Data Parallel was accidentally stored back in 2017.
            model = nn.DataParallel(KneeNet(64, 0.2, False)).to(self.device)
            model.load_state_dict(state)
            self.logger.log(logging.INFO, f'Model {idx} state has been loaded')
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
            self.logger.log(logging.INFO, f'Model {idx} has been initialized')

        self.__dict__['_modules'].update(models)
        self.to(self.device)
        self.logger.log(logging.INFO, f'The whole pipeline has been moved to {self.device}')

    def load_picture(self, fname, nbits=16, flip_left=False):
        """

        :param fname: str or numpy.ndarray
            Takes either full path to the image or the numpy array
        :return:
        """
        self.logger.log(logging.DEBUG, f'Processing {nbits} bit {"left" if flip_left else "right"} image')
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
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        img_cropped = self.cropper(img)
        lateral, medial = get_pair(img_cropped)

        lateral = self.patch_transform(lateral).to(self.device)
        medial = self.patch_transform(medial).to(self.device)
        self.logger.log(logging.DEBUG, f'Image pre-processing has been finished')
        return img_cropped, lateral.view(1, 1, 128, 128), medial.view(1, 1, 128, 128)

    def decompose_forward_avg(self, net, l, m):
        # Reducing the memory footprint.
        # We don't really need gradients to compute the features
        self.logger.log(logging.INFO, f'Forward pass started for {hex(id(net))}')
        with torch.no_grad():
            l_o = net.branch(l)
            m_o = net.branch(m)
            l_o_avg = F.adaptive_avg_pool2d(l_o, (1, 1))
            m_o_avg = F.adaptive_avg_pool2d(m_o, (1, 1))
        self.logger.log(logging.DEBUG, f'Features have been extracted')
        # These variables will requre features as they will initiate the forward pass to the FC layer
        # From which we will get the gradients
        l_o_avg.requires_grad = True
        m_o_avg.requires_grad = True
        # A normal forward pass. Concatenating the outputs from the lateral and the medial sides
        self.logger.log(logging.DEBUG, f'Pushing the feature maps through FC layer')
        concat = torch.cat([l_o_avg, m_o_avg], 1)
        # Passing the results through an FC layer
        o = net.final(concat.view(l.size(0), net.final.in_features))
        self.logger.log(logging.INFO, f'Model {hex(id(net))} finished predictions')
        return l_o, m_o, l_o_avg, m_o_avg, o

    def weigh_maps(self, weights, maps):
        maps = maps.squeeze()
        weights = weights.squeeze()

        res = torch.zeros(maps.size()[-2:]).to(self.device)

        for i, w in enumerate(weights):
            res += w * maps[i]
        return res

    def extract_gradcam_weighted_maps(self, o_l, o_m, wl, wm):
        self.logger.log(logging.DEBUG, f'GradCAM-based weighing started')
        # After extracting the features, we weigh them based on the provided weights
        o_l = self.weigh_maps(wl, o_l)
        o_m = self.weigh_maps(wm, o_m)
        return F.relu(o_l), F.relu(o_m)

    def compute_gradcam(self, features, img_size, ps, smoothing=7):
        self.logger.log(logging.INFO, f'GradCAM computation has been started')
        w_lateral, w_medial = self.grads_l1[0].data, self.grads_m1[0].data
        ol1, om1 = self.extract_gradcam_weighted_maps(features['net1'][0], features['net1'][1], w_lateral, w_medial)

        w_lateral, w_medial = self.grads_l2[0].data, self.grads_m2[0].data
        ol2, om2 = self.extract_gradcam_weighted_maps(features['net2'][0], features['net2'][1], w_lateral, w_medial)

        w_lateral, w_medial = self.grads_l3[0].data, self.grads_m3[0].data
        ol3, om3 = self.extract_gradcam_weighted_maps(features['net3'][0], features['net3'][1], w_lateral, w_medial)

        l_out = (ol1 + ol2 + ol3) / 3.
        m_out = (om1 + om2 + om3) / 3.
        self.logger.log(logging.INFO, f'Creating the heatmap')
        heatmap = inverse_pair_mapping(l_out.detach().to('cpu').numpy(),
                                       np.fliplr(m_out.detach().to('cpu').numpy()),
                                       img_size, ps, smoothing)
        heatmap -= heatmap.min()
        heatmap /= heatmap.max()
        return heatmap

    def forward(self, l, m):
        self.logger.log(logging.INFO, f'Forward pass started')
        self.empty_gradient_arrays()

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
        self.logger.log(logging.INFO, f'Prediction started')
        if fname_suffix is not None:
            pass
        elif isinstance(fileobj_in, str):
            fname_suffix = os.path.splitext(os.path.basename(fileobj_in))[0]
        else:
            fname_suffix = ''

        img, heatmap, probs = self.predict(x=fileobj_in, nbits=nbits, flip_left=flip_left)
        self.logger.log(logging.INFO, f'Drawing the heatmap')
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

        # Making a bar plot for displaying probabilities
        self.logger.log(logging.INFO, f'Drawing the vector with probabilities')
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
        self.logger.log(logging.INFO, f'Sending the results back to the user')
        return img, img_overlayed, probs_bar, probs.squeeze().argmax()

    def request_landmarks(self, kneel_addr, file):
        self.logger.log(logging.INFO, f'Sending the image to KNEEL: {os.environ["KNEEL_ADDR"]}')
        if kneel_addr is None:
            kneel_addr = os.environ["KNEEL_ADDR"]

        response = requests.post(f'{kneel_addr}/kneel/predict/bilateral', json=file)
        landmarks = response.json()
        return landmarks

    def localize_bilateral(self, dicom_raw, sizemm, pad, kneel_addr=None, landmarks=None):
        if landmarks is None:
            landmarks = self.request_landmarks(kneel_addr, {'dicom': base64.b64encode(dicom_raw).decode()})

        if landmarks['R'] is None:
            self.logger.log(logging.INFO, f'Landmarks are not found. Returning None')
            return None

        self.logger.log(logging.INFO, f'Image decoding and pre-processing started')
        raw = DicomBytesIO(dicom_raw)
        dicom_data = dcmread(raw)
        img, spacing, dicom_data = read_dicom(dicom_data)
        img = process_xray(img, 5, 99, 255).astype(np.uint8)
        sizepx = int(np.round(sizemm / spacing))
        self.logger.log(logging.DEBUG, f'Padding the image')
        row, col = img.shape
        tmp = np.zeros((row + 2 * pad, col + 2 * pad))
        tmp[pad:pad + row, pad:pad + col] = img
        img = tmp

        landmarks_l = np.array(landmarks['L']) + pad
        landmarks_r = np.array(landmarks['R']) + pad
        # Extracting center landmarks
        lcx, lcy = landmarks_l[4]
        rcx, rcy = landmarks_r[4]

        img_left = img[(lcy - sizepx // 2):(lcy + sizepx // 2),
                   (lcx - sizepx // 2):(lcx + sizepx // 2)].astype(np.uint8)

        img_right = img[(rcy - sizepx // 2):(rcy + sizepx // 2),
                    (rcx - sizepx // 2):(rcx + sizepx // 2)].astype(np.uint8)

        self.logger.log(logging.INFO, f'Returning localized left and right knees')
        return img_left, img_right

    def predict_draw_bilateral(self, dicom_raw, sizemm, pad, kneel_addr=None, landmarks=None):
        res_landmarks = self.localize_bilateral(dicom_raw, sizemm, pad, kneel_addr, landmarks)
        if res_landmarks is None:
            return None

        img_left, img_right = res_landmarks

        img_l, img_hm_l, preds_bar_l, pred_l = self.predict_draw(fileobj_in=img_left,
                                                                 nbits=8,
                                                                 path_dir_out=None,
                                                                 flip_left=True)

        img_r, img_hm_r, preds_bar_r, pred_r = self.predict_draw(fileobj_in=img_right,
                                                                 nbits=8,
                                                                 path_dir_out=None,
                                                                 flip_left=False)
        return img_l, img_hm_l, preds_bar_l, pred_l, img_r, img_hm_r, preds_bar_r, pred_r


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

    net = KneeNetEnsemble(glob.glob(os.path.join(args.snapshots_path, "*", '*.pth')),
                          mean_std_path=os.path.join(args.snapshots_path, 'mean_std.npy'),
                          device=args.device)

    paths_test_files = glob.glob(os.path.join(args.images, '*.png'))

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.output_csv, 'w') as f:
        f.write('IMG,KL_R,KL_L\n')
        for path_test_file in tqdm(paths_test_files, total=len(paths_test_files)):
            with open(path_test_file, 'rb') as fdicom:
                dicom_raw_local = fdicom.read()

            res_bilateral = net.predict_draw_bilateral(dicom_raw_local, 140, 300)
            if res_bilateral is None:
                print('Could not localize the landmarks!')
            img_l, img_hm_l, preds_bar_l, pred_l, img_r, img_hm_r, preds_bar_r, pred_r = res_bilateral

            line = '{},{},{}\n'.format(path_test_file.split('/')[-1], pred_r, pred_l)
            f.write(line)
