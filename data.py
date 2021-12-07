import h5py
import torch
import numpy as np

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

total_num = {
    'vgg': 200,
    'dcc': 176,
    'mbm': 44,
    'adi': 200, }

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset: str, seed: int, train: bool, augment: bool, val_percent: float):
        super(Dataset, self).__init__()
        if dataset not in ['vgg', 'dcc', 'mbm', 'adi']:
            raise ValueError(f'Wrong dataset name: {dataset}')
        self.h5 = h5py.File(data_path[dataset], 'r')
        self.seed = seed
        self.dataset = dataset
        self.image_shape = image_shape[dataset]
        self.val_percent = val_percent
        
        np.random.seed(seed)
        inds = np.linspace(0, total_num[dataset] - 1, total_num[dataset]).astype('uint8')
        val = np.delete(inds, np.random.choice(inds, int(total_num[dataset] * (1 - val_percent)), replace=False))
        
        if train:
            inds = np.setdiff1d(inds, val) 
        else:
            inds = val    
        
        with h5py.File(data_path[dataset], 'r') as hf:
            self.images = hf.get('imgs')[inds].astype('float32')
            self.labels = hf.get('counts')[inds] * 100.
        
        self.augment = augment
        self.train = train

    # !!! make faster
    def preprocess(self, image, label):
        def _augment(D):
            if self.image_shape == (600, 600, 3):
                final = 576
                x_crop = np.random.randint(0, 24) 
                y_crop = np.random.randint(0, 24) 
                D = D[x_crop:(final+x_crop),y_crop:(final+y_crop),:]
            elif self.image_shape == (152, 152, 3):
                final = 144
                x_crop = np.random.randint(0, 6)
                y_crop = np.random.randint(0, 6)
                D = D[x_crop:(final+x_crop),y_crop:(final+y_crop),:]
            elif self.image_shape == (256, 256, 3):
                final = 224
                x_crop = np.random.randint(0, 32)
                y_crop = np.random.randint(0, 32)
                D = D[x_crop:(final+x_crop),y_crop:(final+y_crop),:]
            else:
                raise ValueError('Incorrect dataset')   
           
            r = np.random.randint(2)
            D = np.flip(D, axis=r)
            r = np.random.randint(2)
            D = np.flip(D, axis=r) 
                
            return D         


        np.random.seed(self.seed)
        inds = np.linspace(0, total_num[self.dataset] - 1, total_num[self.dataset]).astype('uint8')
        val = np.delete(inds, np.random.choice(inds, int(total_num[self.dataset] * (1 - self.val_percent)), replace=False))
        inds = np.setdiff1d(inds, val)
        
        with h5py.File(data_path[self.dataset], 'r') as hf:
            train_images = hf.get('imgs')[inds].astype('float32')
            train_labels = hf.get('counts')[inds] * 100.
        
        mn = np.mean(train_images, axis=(0, 1, 2))
        std = np.std(train_images, axis=(0, 1, 2)) 

        image = (image - mn) / std
        label = label.astype(np.float32)[..., np.newaxis] 

        D = np.concatenate((image, label), axis=-1) 

        if self.augment:
            D = _augment(D)
        
        
        return D[:, :, :3], D[:, :, 3:]
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        im, lb = self.preprocess(self.images[index], self.labels[index])
        return im.transpose(), lb.transpose()
