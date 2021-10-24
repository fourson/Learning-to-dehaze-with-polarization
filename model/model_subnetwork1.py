import functools

import torch
import torch.nn as nn

from base.base_model import BaseModel

from .layer_utils.unet import UnetBackbone
from .layer_utils.resnet import ResnetBackbone
from .layer_utils.funcs import get_norm_layer


class DefaultModel(BaseModel):
    """
        Define the baseline network to predict P_A, P_T and T
    """

    def __init__(self, init_dim=32, norm_type='instance', use_dropout=False, C=3, residual=False):
        super(DefaultModel, self).__init__()
        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.feature_extraction1 = nn.Sequential(
            nn.Conv2d(3 * C, init_dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(init_dim, init_dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(init_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(init_dim, init_dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.backbone1 = UnetBackbone(init_dim, output_nc=init_dim, n_downsampling=4, use_conv_to_downsample=True,
                                      norm_type=norm_type, use_dropout=use_dropout, mode='default')
        self.out_block1_1 = nn.Sequential(
            nn.Conv2d(init_dim, C, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

        self.out_block1_2 = nn.Sequential(
            nn.Conv2d(init_dim, C, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

        self.feature_extraction2 = nn.Sequential(
            nn.Conv2d(4 * C, init_dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(init_dim, init_dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(init_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(init_dim, init_dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.backbone2 = ResnetBackbone(init_dim, output_nc=init_dim, n_downsampling=3, n_blocks=5,
                                        norm_type=norm_type, use_dropout=use_dropout)

        self.residual = residual
        if self.residual:
            self.out_block2 = nn.Sequential(
                nn.Conv2d(init_dim, C, kernel_size=1, stride=1),
                nn.Tanh()
            )
        else:
            self.out_block2 = nn.Sequential(
                nn.Conv2d(init_dim, C, kernel_size=1, stride=1),
                nn.Sigmoid()
            )

    def forward(self, I_alpha, I, delta_I):
        # |input:
        #  I_alpha: three polarized images, [0, 1], as float32
        #  I: total light, [0, 1] float, as float32
        #  delta_I: the PD (polarization difference) image, [0, 1] float, as float32 (delta_I = P * I)
        # |output:
        #  P_A: airlight degree of polarization, [0, 1] float, as float32
        #  P_T: transmission light degree of polarization, [0, 1] float, as float32
        #  T: transmission light, [0, 1] float, as float32

        feature1 = self.feature_extraction1(I_alpha)
        backbone_out1 = self.backbone1(feature1)
        P_A = self.out_block1_1(backbone_out1)
        P_T = self.out_block1_2(backbone_out1)

        # # for spatially-uniform model
        # P_A_mean = torch.mean(P_A, (2, 3), keepdim=True)
        # P_A = torch.ones_like(P_A) * P_A_mean
        # P_T_mean = torch.mean(P_T, (2, 3), keepdim=True)
        # P_T = torch.ones_like(P_T) * P_T_mean

        T_hat = torch.clamp((delta_I - I * P_A) / (P_T - P_A + 1e-7), min=0, max=1)
        # T_hat = torch.clamp(I - (delta_I) / (P_A + 1e-7), min=0, max=1)  # for ablation P_T_zero
        cat = torch.cat([T_hat, I_alpha], dim=1)
        feature2 = self.feature_extraction2(cat)
        backbone_out2 = self.backbone2(feature2)
        if self.residual:
            T = self.out_block2(backbone_out2) + T_hat
        else:
            T = self.out_block2(backbone_out2)

        # T = T_hat  # without T refinement

        return P_A, P_T, T
        # use this to output T_hat for visualization
        # return P_A, P_T, T, T_hat


class DirectTModel(BaseModel):
    """
        predict T directly
    """
    def __init__(self, init_dim=32, norm_type='instance', use_dropout=False, C=3):
        super(DirectTModel, self).__init__()
        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3 * C, init_dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(init_dim, init_dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(init_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(init_dim, init_dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(init_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.backbone = UnetBackbone(init_dim, output_nc=init_dim, n_downsampling=4, use_conv_to_downsample=True,
                                     norm_type=norm_type, use_dropout=use_dropout, mode='default')
        self.out_block = nn.Sequential(
            nn.Conv2d(init_dim, C, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, I_alpha, I, delta_I):
        # |input:
        #  I_alpha: three polarized images, [0, 1], as float32
        #  I: total light, [0, 1] float, as float32
        #  delta_I: the PD (polarization difference) image, [0, 1] float, as float32 (delta_I = P * I)
        # |output:
        #  T: transmission light, [0, 1] float, as float32

        feature = self.feature_extraction(I_alpha)
        backbone_out = self.backbone(feature)
        T = self.out_block(backbone_out)

        P_A = torch.zeros_like(T)
        P_T = torch.zeros_like(T)

        return P_A, P_T, T
