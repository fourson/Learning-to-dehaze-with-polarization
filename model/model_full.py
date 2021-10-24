import functools

import torch
import torch.nn as nn

from base.base_model import BaseModel

from .layer_utils.unet import UnetBackbone
from .layer_utils.resnet import ResnetBackbone
from .layer_utils.funcs import get_norm_layer

from model.model_subnetwork1 import DefaultModel as T_model
from model.model_subnetwork2 import DefaultModel as R_model

from model.model_subnetwork1 import DirectTModel as Direct_T_model
from model.model_subnetwork2 import DirectRModel as Direct_R_model


class DefaultModel(BaseModel):
    """
        Define the baseline network to predict P_A, P_T, T, A_infinity and R
    """

    def __init__(self, init_dim=32, norm_type='instance', use_dropout=False, C=3, residual=False):
        super(DefaultModel, self).__init__()

        self.T_model = T_model(init_dim=init_dim, norm_type=norm_type, use_dropout=use_dropout, C=C, residual=residual)
        self.R_model = R_model(init_dim=init_dim, norm_type=norm_type, use_dropout=use_dropout, C=C)

    def forward(self, I_alpha, I, delta_I):
        # |input:
        #  I_alpha: three polarized images, [0, 1], as float32
        #  I: total light, [0, 1] float, as float32
        #  delta_I: the PD (polarization difference) image, [0, 1] float, as float32 (delta_I = P * I)
        # |output:
        #  P_A: airlight degree of polarization, [0, 1] float, as float32
        #  P_T: transmission light degree of polarization, [0, 1] float, as float32
        #  T: transmission light, [0, 1] float, as float32
        #  A_infinity: airlight radiance corresponding to an object at an infinite distance, [0, 1] float, as float32
        #  R: radiance, [0, 1] float, as float32

        P_A, P_T, T = self.T_model(I_alpha, I, delta_I)
        A_infinity, R = self.R_model(I_alpha, I, T)
        return P_A, P_T, T, A_infinity, R
        # use this to output T_hat for visualization
        # P_A, P_T, T, T_hat = self.T_model(I_alpha, I, delta_I)
        # A_infinity, R, R_hat = self.R_model(I_alpha, I, T)
        # return P_A, P_T, T, A_infinity, R, T_hat, R_hat


class DirectTRModel(BaseModel):
    def __init__(self, init_dim=32, norm_type='instance', use_dropout=False, C=3, residual=False):
        super(DirectTRModel, self).__init__()

        self.Direct_T_model = Direct_T_model(init_dim=init_dim, norm_type=norm_type, use_dropout=use_dropout, C=C, residual=residual)
        self.Direct_R_model = Direct_R_model(init_dim=init_dim, norm_type=norm_type, use_dropout=use_dropout, C=C)

    def forward(self, I_alpha, I, delta_I):
        # |input:
        #  I_alpha: three polarized images, [0, 1], as float32
        #  I: total light, [0, 1] float, as float32
        #  delta_I: the PD (polarization difference) image, [0, 1] float, as float32 (delta_I = P * I)
        # |output:
        #  R: radiance, [0, 1] float, as float32

        P_A, P_T, T = self.Direct_T_model(I_alpha, I, delta_I)
        A_infinity, R = self.Direct_R_model(I_alpha, I, T)
        return P_A, P_T, T, A_infinity, R