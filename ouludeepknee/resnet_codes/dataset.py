"""
Dataset classes and samplers


(c) Aleksei Tiulpin, University of Oulu, 2017
"""

import os

import numpy as np
from PIL import Image

import torch
import torch.utils.data as data


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
        img = np.array(img, dtype=float)
        img = np.uint8(255*(img/65535.))
        img = Image.fromarray(np.repeat(img[:, :, np.newaxis], 3, axis=2))

        img = self.transform(img)

        return img, target, fname

    def __len__(self):
        return len(self.names)


class LimitedRandomSampler(data.sampler.Sampler):
    def __init__(self, data_source, nb, bs):
        self.data_source = data_source
        self.n_batches = nb
        self.bs = bs

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).long()[:self.n_batches*self.bs])

    def __len__(self):
        return self.n_batches*self.bs
