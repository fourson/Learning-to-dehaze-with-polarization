import os
import fnmatch

import numpy as np
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """
        for learning R

        as input:
        I_alpha: three polarized images, [0, 1], as float32
        R_hat: radiance, [0, 1] float, as float32

        as target:
        R: radiance, [0, 1] float, as float32
    """

    def __init__(self, data_dir, extra_dir='', transform=None):
        self.I_alpha_dir = os.path.join(data_dir, 'I_alpha')
        if extra_dir:
            self.R_hat_dir = os.path.join(extra_dir, 'R_by_hand')
        else:
            self.R_hat_dir = os.path.join(data_dir, 'R_by_hand')

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
        R_hat = np.load(os.path.join(self.R_hat_dir, self.names[index]))  # [0, 1] float, as float32

        # (3*C, H, W)
        I_alpha = torch.tensor(np.transpose(I_alpha, (2, 0, 1)), dtype=torch.float32)  # [0, 1] float, as float32
        # (C, H, W)
        R_hat = torch.tensor(np.transpose(R_hat, (2, 0, 1)), dtype=torch.float32)  # [0, 1] float, as float32

        # as target:
        # (H, W, C)
        R = np.load(os.path.join(self.R_dir, self.names[index]))  # [0, 1] float, as float32

        # (C, H, W)
        R = torch.tensor(np.transpose(R, (2, 0, 1)), dtype=torch.float32)  # [0, 1] float, as float32

        name = self.names[index].split('.')[0]

        if self.transform:
            R_hat = self.transform(R_hat)

            R = self.transform(R)

        return {'I_alpha': I_alpha, 'R_hat': R_hat, 'R': R, 'name': name}


class InferDataset(Dataset):
    """
        for learning R

        as input:
        I_alpha: three polarized images, [0, 1], as float32
        R_hat: radiance, [0, 1] float, as float32
    """

    def __init__(self, data_dir, extra_dir='', transform=None):
        self.I_alpha_dir = os.path.join(data_dir, 'I_alpha')
        if extra_dir:
            self.R_hat_dir = os.path.join(extra_dir, 'R_by_hand')
        else:
            self.R_hat_dir = os.path.join(data_dir, 'R_by_hand')

        self.names = fnmatch.filter(os.listdir(self.R_hat_dir), '*.npy')

        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # as input:
        # (H, W, 3*C)
        I_alpha = np.load(os.path.join(self.I_alpha_dir, self.names[index]))  # [0, 1] float, as float32
        # (H, W, C)
        R_hat = np.load(os.path.join(self.R_hat_dir, self.names[index]))  # [0, 1] float, as float32

        # (3*C, H, W)
        I_alpha = torch.tensor(np.transpose(I_alpha, (2, 0, 1)), dtype=torch.float32)  # [0, 1] float, as float32
        # (C, H, W)
        R_hat = torch.tensor(np.transpose(R_hat, (2, 0, 1)), dtype=torch.float32)  # [0, 1] float, as float32

        name = self.names[index].split('.')[0]

        if self.transform:
            R_hat = self.transform(R_hat)

        return {'I_alpha': I_alpha, 'R_hat': R_hat, 'name': name}
