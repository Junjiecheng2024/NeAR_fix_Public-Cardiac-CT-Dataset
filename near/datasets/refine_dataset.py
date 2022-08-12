import os
import pandas as pd
from skimage.transform import resize
import numpy as np
import torch
from torch.utils.data import Dataset


class AlanDataset(Dataset):

    def __init__(self, root='../../data/Alan', appearance_path='appearance', shape_path='shape', resolution=128, n_samples=None):

        self.root = root
        self.resolution = resolution
        self.appearance_dir = os.path.join(root, appearance_path)
        self.shape_dir = os.path.join(root, shape_path)

        df = pd.read_csv(os.path.join(self.root, 'info.csv'))
        info = df[df['low_quality'].isnull()]
        self.info = info[['ROI_id', 'ROI_anomaly']]
        self.info.reset_index(drop=True, inplace=True)

        if (n_samples is not None) and len(self.info) > n_samples:
            self.info = self.info[:n_samples]

    def __getitem__(self, index):
        data = self.info.loc[index]['ROI_id']

        shape = np.load(os.path.join(self.shape_dir, data+'.npy'))
        appearance = np.load(os.path.join(self.appearance_dir, data+'.npy'))

        if self.resolution != 128:
            shape = resize(shape.astype(float), (self.resolution, )*3, order=3).astype(bool)
            appearance = resize(appearance.astype(float), (self.resolution, )*3, order=3)

        return (index, torch.tensor(shape).float().unsqueeze_(0), torch.tensor(appearance).float().unsqueeze_(0))

    def __len__(self):
        return len(self.info)


class AbdomenCTDataset(Dataset):

    def __init__(self, root='../../data/AbdomenCT/', appearance_path='appearance', shape_path='shape', resolution=128, n_samples=None):

        self.root = root
        self.resolution = resolution
        self.appearance_dir = os.path.join(root, appearance_path)
        self.shape_dir = os.path.join(root, shape_path)

        df = pd.read_csv(os.path.join(self.root, 'info.csv'))
        self.info = df[['id']]

        if (n_samples is not None) and len(self.info) > n_samples:
            self.info = self.info[:n_samples]

    def __getitem__(self, index):
        data = self.info.loc[index]['id']

        shape = np.load(os.path.join(self.shape_dir, data+'.npy'))
        appearance = np.load(os.path.join(self.appearance_dir, data+'.npy'))

        if self.resolution != 128:
            shape = resize(shape.astype(float), (self.resolution, )*3, order=3).astype(bool)
            appearance = resize(appearance.astype(float), (self.resolution, )*3, order=3)

        return (index, torch.tensor(shape).float().unsqueeze_(0), torch.tensor(appearance).float().unsqueeze_(0))

    def __len__(self):
        return len(self.info)


class AbdomenCTDistortedDataset(Dataset):

    def __init__(self, root='../../data/AbdomenCT/', appearance_path='appearance', golden_shape_path='shape', distort_shape_path='distorted', resolution=128, n_samples=None):

        self.root = root
        self.resolution = resolution
        self.appearance_dir = os.path.join(root, appearance_path)
        self.golden_shape_dir = os.path.join(root, golden_shape_path)
        self.distort_shape_dir = os.path.join(root, distort_shape_path)

        df = pd.read_csv(os.path.join(self.root, 'info.csv'))
        self.info = df[['id']]

        if (n_samples is not None) and len(self.info) > n_samples:
            self.info = self.info[:n_samples]

    def __getitem__(self, index):
        data = self.info.loc[index]['id']

        golden_shape = np.load(os.path.join(self.golden_shape_dir, data+'.npy'))
        appearance = np.load(os.path.join(self.appearance_dir, data+'.npy'))
        distort_shape = np.load(os.path.join(self.distort_shape_dir, data+'.npy'))

        if self.resolution != 128:
            golden_shape = resize(golden_shape.astype(float), (self.resolution, )*3, order=3).astype(bool)
            distort_shape = resize(distort_shape.astype(float), (self.resolution, )*3, order=3).astype(bool)
            appearance = resize(appearance.astype(float), (self.resolution, )*3, order=3)

        return (index, torch.tensor(golden_shape).float().unsqueeze_(0), torch.tensor(distort_shape).float().unsqueeze_(0), torch.tensor(appearance).float().unsqueeze_(0))

    def __len__(self):
        return len(self.info)
