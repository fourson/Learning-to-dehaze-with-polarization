import os
import fnmatch

import numpy as np
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """
        for learning P_A, P_T, T, A_infinity and R

        as input:
        I_alpha: three polarized images, [0, 1], as float32
        I: total light, [0, 1] float, as float32
        delta_I: the PD (polarization difference) image, [0, 1] float, as float32
    """

    def __init__(self, data_dir, extra_dir='', transform=None):
        self.I_alpha_dir = os.path.join(data_dir, 'I_alpha')
        if extra_dir:
            self.I_dir = os.path.join(extra_dir, 'I_hat')
            self.delta_I_dir = os.path.join(extra_dir, 'delta_I_hat')
        else:
            self.I_dir = os.path.join(data_dir, 'I_hat')
            self.delta_I_dir = os.path.join(data_dir, 'delta_I_hat')

        self.P_A_dir = os.path.join(data_dir, 'P_A')
        self.P_T_dir = os.path.join(data_dir, 'P_T')
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
        delta_I = np.load(os.path.join(self.delta_I_dir, self.names[index]))  # [0, 1] float, as float32

        # (3*C, H, W)
        I_alpha = torch.tensor(np.transpose(I_alpha, (2, 0, 1)), dtype=torch.float32)  # [0, 1] float, as float32
        # (C, H, W)
        I = torch.tensor(np.transpose(I, (2, 0, 1)), dtype=torch.float32)  # [0, 1] float, as float32
        delta_I = torch.tensor(np.transpose(delta_I, (2, 0, 1)), dtype=torch.float32)  # [0, 1] float, as float32

        # as target:
        # (C,)
        P_A = np.load(os.path.join(self.P_A_dir, self.names[index]))  # [0, 1] float, as float32
        # (H, W, C)
        P_T = np.load(os.path.join(self.P_T_dir, self.names[index]))  # [0, 1] float, as float32
        T = np.load(os.path.join(self.T_dir, self.names[index]))  # [0, 1] float, as float32
        A_infinity = np.load(os.path.join(self.A_infinity_dir, self.names[index]))  # [0, 1] float, as float32
        R = np.load(os.path.join(self.R_dir, self.names[index]))  # [0, 1] float, as float32

        # (C, H, W)
        P_T = torch.tensor(np.transpose(P_T, (2, 0, 1)), dtype=torch.float32)  # [0, 1] float, as float32
        T = torch.tensor(np.transpose(T, (2, 0, 1)), dtype=torch.float32)  # [0, 1] float, as float32
        P_A = torch.tensor(np.broadcast_to(P_A[:, None, None], P_T.shape).copy(),
                           dtype=torch.float32)  # [0, 1] float, as float32
        A_infinity = torch.tensor(np.transpose(A_infinity, (2, 0, 1)), dtype=torch.float32)  # [0, 1] float, as float32
        R = torch.tensor(np.transpose(R, (2, 0, 1)), dtype=torch.float32)  # [0, 1] float, as float32

        name = self.names[index].split('.')[0]

        if self.transform:
            I_alpha = self.transform(I_alpha)
            I = self.transform(I)
            delta_I = self.transform(delta_I)

            P_A = self.transform(P_A)
            P_T = self.transform(P_T)
            T = self.transform(T)
            A_infinity = self.transform(A_infinity)
            R = self.transform(R)

        return {'I_alpha': I_alpha, 'I': I, 'delta_I': delta_I, 'P_A': P_A, 'P_T': P_T, 'T': T,
                'A_infinity': A_infinity, 'R': R, 'name': name}


class InferDataset(Dataset):
    """
        for learning P_A, P_T, T, A_infinity and R

        as input:
        I_alpha: three polarized images, [0, 1], as float32
        I: total light, [0, 1] float, as float32
        delta_I: the PD (polarization difference) image, [0, 1] float, as float32
    """

    def __init__(self, data_dir, extra_dir='', transform=None):
        self.I_alpha_dir = os.path.join(data_dir, 'I_alpha')
        if extra_dir:
            self.I_dir = os.path.join(extra_dir, 'I_hat')
            self.delta_I_dir = os.path.join(extra_dir, 'delta_I_hat')
        else:
            self.I_dir = os.path.join(data_dir, 'I_hat')
            self.delta_I_dir = os.path.join(data_dir, 'delta_I_hat')

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
        delta_I = np.load(os.path.join(self.delta_I_dir, self.names[index]))  # [0, 1] float, as float32

        # (3*C, H, W)
        I_alpha = torch.tensor(np.transpose(I_alpha, (2, 0, 1)), dtype=torch.float32)  # [0, 1] float, as float32
        # (C, H, W)
        I = torch.tensor(np.transpose(I, (2, 0, 1)), dtype=torch.float32)  # [0, 1] float, as float32
        delta_I = torch.tensor(np.transpose(delta_I, (2, 0, 1)), dtype=torch.float32)  # [0, 1] float, as float32

        name = self.names[index].split('.')[0]

        if self.transform:
            I_alpha = self.transform(I_alpha)
            I = self.transform(I)
            delta_I = self.transform(delta_I)

        return {'I_alpha': I_alpha, 'I': I, 'delta_I': delta_I, 'name': name}
