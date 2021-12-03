import h5py
import os
import numpy as np
import torchvision
from config import config as cfg
from utils import shuffle

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
        
        # normalize training and test sets + mean center
        self.train = (imgs[ind[:self.train_num], ...] - mn) / \
            std, counts[ind[:self.train_num], ...]
        self.test = (imgs[ind[self.train_num:], ...] - mn) / \
            std, counts[ind[self.train_num:], ...]
    
    # TODO: remove batch size and epochs as input parameters?  Or handle
    # them here? 
    def preprocessing(self, training, augment, batch_size, num_epochs):
        def _augment(D):
            '''
            private inner function that assumes the input dataset
            contains both features and labels are included.
            labels are in the last dimensions.
            
            example:

            features.shape: (64, 256, 256, 3)
            labels.shape: (64, 256, 256, 1)
            D.shape: (64, 256, 256, 4)
            
            '''
            if self.image_shape == (600, 600, 3):
                final = 576
                x_crop = np.random.randint(0, 24) 
                y_crop = np.random.randint(0, 24) 
                D = D[:,x_crop:(final+x_crop),y_crop:(final+y_crop),:]
            elif self.image_shape == (152, 152, 3):
                final = 144
                x_crop = np.random.randint(0, 6)
                y_crop = np.random.randint(0, 6)
                D = D[:,x_crop:(final+x_crop),y_crop:(final+y_crop),:]
            elif self.image_shape == (256, 256, 3):
                final = 224
                x_crop = np.random.randint(0, 32)
                y_crop = np.random.randint(0, 32)
                D = D[:,x_crop:(final+x_crop),y_crop:(final+y_crop),:]
            else:
                raise ValueError('Incorrect dataset')   
            
            for i in range(D.shape[0]):
                r = np.random.randint(2)
                D[i,:,:,:] = np.flip(D[i,:,:,:], axis=r)
                r = np.random.randint(2)
                D[i,:,:,:] = np.flip(D[i,:,:,:], axis=r)
                
            return D         

        if training:
            dataset = np.concatenate(self.train, axis=-1)
        else:
            dataset = np.concatenate(self.test, axis=-1)

        if augment:
            dataset = _augment(dataset)
        dataset = shuffle(dataset) 
        #dataset = np.tile(shuffle(dataset), num_epochs)
        
        return dataset[:, :, :, :3], dataset[:, :, :, 3:]


 
