import functools

import torch
import torch.nn as nn

from base.base_model import BaseModel
from .layer_utils.unet import UnetBackbone
from .layer_utils.resnet import ResnetBackbone
from .layer_utils.funcs import get_norm_layer

from model.model_R_refine_model import DefaultModel as R_refine_model


class DefaultModel(BaseModel):
    """
        Define the baseline network to predict A_infinity and R
    """

    def __init__(self, init_dim=32, norm_type='instance', use_dropout=False, C=3):
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
        self.out_block1 = nn.Sequential(
            nn.Conv2d(init_dim, C, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

        # may get better results if we pretrain the R_refine_model in a larger dataset?
        self.R_refine_model = R_refine_model(init_dim=2 * init_dim, norm_type='instance', use_dropout=False, C=3,
                                             residual=False)

    def forward(self, I_alpha, I, T):
        # |input:
        #  I_alpha: three polarized images, [0, 1], as float32
        #  I: total light, [0, 1] float, as float32
        #  T: transmission light, [0, 1] float, as float32
        # |output:
        #  A_infinity: airlight radiance corresponding to an object at an infinite distance, [0, 1] float, as float32
        #  R: radiance, [0, 1] float, as float32

        feature1 = self.feature_extraction1(I_alpha)
        backbone_out1 = self.backbone1(feature1)
        A_infinity = self.out_block1(backbone_out1)

        # # for spatially-uniform model
        # A_infinity_mean = torch.mean(A_infinity, (2, 3), keepdim=True)
        # A_infinity = torch.ones_like(A_infinity) * A_infinity_mean

        R_hat = torch.clamp((T * A_infinity + 1e-7) / (A_infinity - I + T + 1e-7), min=0, max=1)
        # R_hat.detach() # uncomment this if you want to prevent the back propagation of gradient
        R = self.R_refine_model(I_alpha, R_hat)

        # R = R_hat  # without R refinement

        return A_infinity, R
        # use this to output T_hat for visualization
        # return A_infinity, R, R_hat


class DirectRModel(BaseModel):
    """
        predict R directly
    """
    def __init__(self, init_dim=32, norm_type='instance', use_dropout=False, C=3, residual=False):
        super(DirectRModel, self).__init__()
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
        self.feature_extraction_T = nn.Sequential(
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

        self.out_block = nn.Sequential(
            nn.Conv2d(init_dim, C, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, I_alpha, I, T):
        # |input:
        #  I_alpha: three polarized images, [0, 1], as float32
        #  I: total light, [0, 1] float, as float32
        #  T: transmission light, [0, 1] float, as float32
        # |output:
        # R: radiance, [0, 1] float, as float32

        feature_I_alpha = self.feature_extraction_I_alpha(I_alpha)
        feature_T = self.feature_extraction_T(T)
        cat = torch.cat([feature_I_alpha, feature_T], dim=1)
        backbone_out = self.backbone(cat)
        R = self.out_block(backbone_out)

        A_infinity = torch.zeros_like(R)

        return A_infinity, R
