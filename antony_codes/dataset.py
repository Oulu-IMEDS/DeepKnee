"""
Dataset classes and samplers


(c) Aleksei Tiulpin, University of Oulu, 2017

"""

import torch.utils.data as data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
from PIL import Image,ImageEnhance
import os

import torchvision.transforms as transforms


class KneeGradingDataset(data.Dataset):
    def __init__(self, dataset, split, transform, stage='train'):
        self.dataset = dataset
        self.names = split
        self.transform = transform
        self.stage=stage

    def __getitem__(self, index):
        fname = os.path.join(self.dataset, self.stage, self.names[index])
        target = int(fname.split('/')[-1].split('_')[1])

        if self.stage == 'train':
            fname = os.path.join(self.dataset, self.stage, str(target), self.names[index])
        
        img = Image.open(fname)
        # We will use 8bit 
        tmp = np.array(img, dtype=float)
        img = Image.fromarray(np.uint8(255*(tmp/65535.)))

        img = self.transform(img)

        return img, target, fname

    def __len__(self):
        return len(self.names)



class LimitedRandomSampler(data.sampler.Sampler):
    """
    Allows to have a fixed number of epochs

    """
    def __init__(self, data_source, nb, bs):
        self.data_source = data_source
        self.n_batches = nb
        self.bs = bs

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).long()[:self.n_batches*self.bs])

    def __len__(self):
        return self.n_batches*self.bs


