from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
from skimage.transform import resize


class AlanSegDataset(Dataset):

    def __init__(self, root='../data/Alan', appearance_path='appearance', shape_path='shape', resolution=128, n_samples=None, copy_channels=True):

        self.root = root
        self.resolution = resolution
        self.appearance_dir = os.path.join(root, appearance_path)
        self.shape_dir = os.path.join(root, shape_path)

        df = pd.read_csv(os.path.join(self.root, 'info.csv'))

        info = df[df['low_quality'].isnull()]
        self.info = info[['ROI_id', 'ROI_anomaly']]
        self.info.reset_index(drop=True, inplace=True)

        self.transform = Transform(resolution, copy_channels)

        if (n_samples is not None) and len(self.info) > n_samples:
            self.info = self.info[:n_samples]

    def __getitem__(self, index):
        data = self.info.loc[index]['ROI_id']

        shape = np.load(os.path.join(self.shape_dir, data+'.npy'))
        appearance = np.load(os.path.join(self.appearance_dir, data+'.npy'))

        return self.transform(appearance, shape)

    def __len__(self):
        return len(self.info)


class DistortedSegDataset(Dataset):

    def __init__(self, root='../data/AbdomenCT/', appearance_path='appearance', golden_shape_path='shape', distort_shape_path='distorted', resolution=128, n_samples=None, copy_channels=True):

        self.root = root
        self.resolution = resolution
        self.appearance_dir = os.path.join(root, appearance_path)
        self.golden_shape_dir = os.path.join(root, golden_shape_path)
        self.distort_shape_dir = os.path.join(root, distort_shape_path)

        self.transform = NoiseTransform(resolution, copy_channels)

        df = pd.read_csv(os.path.join(self.root, 'info.csv'))
        self.info = df[['id']]

        if (n_samples is not None) and len(self.info) > n_samples:
            self.info = self.info[:n_samples]

    def __getitem__(self, index):
        data = self.info.loc[index]['id']

        golden_shape = np.load(os.path.join(self.golden_shape_dir, data+'.npy'))
        appearance = np.load(os.path.join(self.appearance_dir, data+'.npy'))
        distort_shape = np.load(os.path.join(self.distort_shape_dir, data+'.npy'))

        return self.transform(appearance, golden_shape, distort_shape)

    def __len__(self):
        return len(self.info)


class NoiseTransform:
    def __init__(self, size, copy_channels=True):
        self.size = size
        self.copy_channels = copy_channels

    def __call__(self, voxel, clean_seg, noise_seg):
        shape = voxel.shape[0]

        if self.size != shape:
            clean_seg = resize(clean_seg.astype(float), (self.size, )*3, order=3) > 0.5
            noise_seg = resize(noise_seg.astype(float), (self.size, )*3, order=3) > 0.5
            voxel = resize(voxel.astype(float), (self.size, )*3, order=3)
            
        if self.copy_channels:
            return np.stack([voxel,voxel,voxel],0).astype(np.float32), \
                    np.expand_dims(clean_seg,0).astype(np.float32), \
                    np.expand_dims(noise_seg,0).astype(np.float32)

        else:
            return np.expand_dims(voxel, 0).astype(np.float32), \
                    np.expand_dims(clean_seg,0).astype(np.float32), \
                    np.expand_dims(noise_seg,0).astype(np.float32)


class Transform:
    def __init__(self, size, copy_channels=True):
        self.size = size
        self.copy_channels = copy_channels

    def __call__(self, voxel, seg):
        shape = voxel.shape[0]

        if self.size != shape:
            seg = resize(seg.astype(float), (self.size, )*3, order=3) > 0.5
            voxel = resize(voxel.astype(float), (self.size, )*3, order=3)
            
        if self.copy_channels:
            return np.stack([voxel,voxel,voxel],0).astype(np.float32), \
                    np.expand_dims(seg,0).astype(np.float32)
        else:
            return np.expand_dims(voxel, 0).astype(np.float32), \
                    np.expand_dims(seg,0).astype(np.float32)
