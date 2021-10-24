import functools

import torch
import torch.nn as nn

from base.base_model import BaseModel
from .layer_utils.unet import UnetBackbone
from .layer_utils.resnet import ResnetBackbone
from .layer_utils.funcs import get_norm_layer


class DefaultModel(BaseModel):
    """
        Define the baseline network to predict R
    """

    def __init__(self, init_dim=64, norm_type='instance', use_dropout=False, C=3, residual=False):
        super(DefaultModel, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        dim = init_dim // 2
        self.feature_extraction_I_alpha = nn.Sequential(
            nn.Conv2d(3 * C, dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(dim),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.feature_extraction_R_hat = nn.Sequential(
            nn.Conv2d(C, dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.backbone = ResnetBackbone(init_dim, output_nc=init_dim, n_downsampling=3, n_blocks=5, norm_type=norm_type,
                                       use_dropout=use_dropout)

        self.residual = residual
        if self.residual:
            self.out_block = nn.Sequential(
                nn.Conv2d(init_dim, C, kernel_size=1, stride=1),
                nn.Tanh()
            )
        else:
            self.out_block = nn.Sequential(
                nn.Conv2d(init_dim, C, kernel_size=1, stride=1),
                nn.Sigmoid()
            )

    def forward(self, I_alpha, R_hat):
        # |input:
        #  I_alpha: three polarized images, [0, 1], as float32
        #  R_hat: radiance, [0, 1] float, as float32
        # |output:
        # R: refined radiance, [0, 1] float, as float32

        feature_I_alpha = self.feature_extraction_I_alpha(I_alpha)
        feature_R_hat = self.feature_extraction_R_hat(R_hat)
        cat = torch.cat([feature_I_alpha, feature_R_hat], dim=1)
        backbone_out = self.backbone(cat)
        if self.residual:
            R = self.out_block(backbone_out) + R_hat
        else:
            R = self.out_block(backbone_out)

        return R
