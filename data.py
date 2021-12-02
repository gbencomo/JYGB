import h5py
import os
import numpy as np
import torch
from config import config as cfg

data_path = {
    'vgg': 'data/VGG/VGG.hdf5',
    'dcc': 'data/DCC/DCC.hdf5',
    'mbm': 'data/MBM/MBM.hdf5',
    'adi': 'data/ADI/ADI.hdf5', }

image_shape = {
    'vgg': (256, 256, 3),
    'dcc': (256, 256, 3),
    'mbm': (600, 600, 3),
    'adi': (152, 152, 3), }

train_num = {
    'vgg': 64,
    'dcc': 100,
    'mbm': 15,
    'adi': 50, }

total_num = {
    'vgg': 200,
    'dcc': 176,
    'mbm': 44,
    'adi': 200, }


class Dataset(object):
    def __init__(self, dataset, seed):
        if dataset not in ['vgg', 'dcc', 'mbm', 'adi']:
            raise ValueError(f'Wrong dataset name: {dataset}')
        self.dataset = dataset
        self.image_shape = image_shape[dataset]
        self.seed = seed
        self.total_num = total_num[self.dataset]
        self.train_num = train_num[self.dataset]

        self.data = {}
        with h5py.File(data_path[dataset], 'r') as hf:
            self.data['imgs'] = hf.get('imgs')[()]
            self.data['counts'] = hf.get('counts')[()]
        self.label_shape = self.image_shape[:2] + (1,)

        imgs = self.data['imgs'].astype(np.float32)
        counts = self.data['counts'].astype(np.float32)[..., np.newaxis]

        imgs = imgs / 255.
        assert np.max(imgs) <= 1
        assert np.min(imgs) >= 0

        assert imgs.shape == (self.total_num,) + self.image_shape
        assert counts.shape == (self.total_num,) + self.label_shape

        np.random.seed(self.seed)
        ind = np.random.permutation(self.total_num)

        mn = np.mean(imgs[ind[:self.train_num], ...], axis=(0, 1, 2))
        std = np.std(imgs[ind[:self.train_num], ...], axis=(0, 1, 2))
        self.train = (imgs[ind[:self.train_num], ...] - mn) / \
            std, counts[ind[:self.train_num], ...]
        self.test = (imgs[ind[self.train_num:], ...] - mn) / \
            std, counts[ind[self.train_num:], ...]


