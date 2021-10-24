import os
import fnmatch

import numpy as np
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """
        for learning A_infinity and R

        as input:
        I_alpha: three polarized images, [0, 1], as float32
        I: total light, [0, 1] float, as float32
        T: transmission light, [0, 1] float, as float32

        as target:
        A_infinity: airlight radiance corresponding to an object at an infinite distance, [0, 1] float, as float32
        R: radiance, [0, 1] float, as float32
    """

    def __init__(self, data_dir, extra_dir='', transform=None):
        self.I_alpha_dir = os.path.join(data_dir, 'I_alpha')
        self.I_dir = os.path.join(data_dir, 'I_hat')
        if extra_dir:
            self.T_dir = os.path.join(extra_dir, 'T')
        else:
            self.T_dir = os.path.join(data_dir, 'T')

        self.A_infinity_dir = os.path.join(data_dir, 'A_infinity')
        self.R_dir = os.path.join(data_dir, 'R')

        self.names = fnmatch.filter(os.listdir(self.I_alpha_dir), '*.npy')

        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # as input:
        # (H, W, 3*C)
        I_alpha = np.load(os.path.join(self.I_alpha_dir, self.names[index]))  # [0, 1] float, as float32
        # (H, W, C)
        I = np.load(os.path.join(self.I_dir, self.names[index]))  # [0, 1] float, as float32
        T = np.load(os.path.join(self.T_dir, self.names[index]))  # [0, 1] float, as float32

        # (3*C, H, W)
        I_alpha = torch.tensor(np.transpose(I_alpha, (2, 0, 1)), dtype=torch.float32)  # [0, 1] float, as float32
        # (C, H, W)
        I = torch.tensor(np.transpose(I, (2, 0, 1)), dtype=torch.float32)  # [0, 1] float, as float32
        T = torch.tensor(np.transpose(T, (2, 0, 1)), dtype=torch.float32)  # [0, 1] float, as float32

        # as target:
        # (H, W, C)
        A_infinity = np.load(os.path.join(self.A_infinity_dir, self.names[index]))  # [0, 1] float, as float32
        R = np.load(os.path.join(self.R_dir, self.names[index]))  # [0, 1] float, as float32

        # (C, H, W)
        A_infinity = torch.tensor(np.transpose(A_infinity, (2, 0, 1)), dtype=torch.float32)  # [0, 1] float, as float32
        R = torch.tensor(np.transpose(R, (2, 0, 1)), dtype=torch.float32)  # [0, 1] float, as float32

        name = self.names[index].split('.')[0]

        if self.transform:
            I_alpha = self.transform(I_alpha)
            I = self.transform(I)
            T = self.transform(T)

            A_infinity = self.transform(A_infinity)
            R = self.transform(R)

        return {'I_alpha': I_alpha, 'I': I, 'T': T, 'A_infinity': A_infinity, 'R': R, 'name': name}


class InferDataset(Dataset):
    """
        for learning A_infinity and R

        as input:
        I_alpha: three polarized images, [0, 1], as float32
        I: total light, [0, 1] float, as float32
        T: transmission light, [0, 1] float, as float32
    """

    def __init__(self, data_dir, extra_dir='', transform=None):
        self.I_alpha_dir = os.path.join(data_dir, 'I_alpha')
        self.I_dir = os.path.join(data_dir, 'I_hat')
        if extra_dir:
            self.T_dir = os.path.join(extra_dir, 'T_hat')
        else:
            self.T_dir = os.path.join(data_dir, 'T_hat')

        self.names = fnmatch.filter(os.listdir(self.I_alpha_dir), '*.npy')

        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # as input:
        # (H, W, 3*C)
        I_alpha = np.load(os.path.join(self.I_alpha_dir, self.names[index]))  # [0, 1] float, as float32
        # (H, W, C)
        I = np.load(os.path.join(self.I_dir, self.names[index]))  # [0, 1] float, as float32
        T = np.load(os.path.join(self.T_dir, self.names[index]))  # [0, 1] float, as float32

        # (3*C, H, W)
        I_alpha = torch.tensor(np.transpose(I_alpha, (2, 0, 1)), dtype=torch.float32)  # [0, 1] float, as float32
        # (C, H, W)
        I = torch.tensor(np.transpose(I, (2, 0, 1)), dtype=torch.float32)  # [0, 1] float, as float32
        T = torch.tensor(np.transpose(T, (2, 0, 1)), dtype=torch.float32)  # [0, 1] float, as float32

        name = self.names[index].split('.')[0]

        if self.transform:
            I_alpha = self.transform(I_alpha)
            I = self.transform(I)
            T = self.transform(T)

        return {'I_alpha': I_alpha, 'I': I, 'T': T, 'name': name}
