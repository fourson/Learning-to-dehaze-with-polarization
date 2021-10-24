import functools

import torch.nn as nn

from .funcs import get_norm_layer


class ResnetBlock(nn.Module):
    """
        Resnet block using bottleneck structure
        dim -> dim
    """

    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()

        sequence = [
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(dim),
            nn.LeakyReLU(0.1, inplace=True)
        ]
        if use_dropout:
            sequence += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = x + self.model(x)
        return out


class DownsamplingBackbone(nn.Module):
    """
        Resnet backbone
        N, input_nc, H, W -> N, init_nc*(2**n_downsampling), 1, 1
    """

    def __init__(self, input_nc, init_nc=64, n_downsampling=4, n_blocks=3, norm_type='instance', use_dropout=False):
        super(DownsamplingBackbone, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        sequence = [
            nn.Conv2d(input_nc, init_nc, kernel_size=7, stride=1, padding=3, bias=use_bias),
            norm_layer(init_nc),
            nn.LeakyReLU(0.1, inplace=True)
        ]

        dim = init_nc
        for i in range(n_downsampling):  # downsample the feature map
            sequence += [
                nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(2 * dim),
                nn.LeakyReLU(0.1, inplace=True),
            ]
            dim *= 2

        for i in range(n_blocks):  # ResBlock
            sequence += [
                ResnetBlock(dim, norm_layer, use_dropout, use_bias)
            ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        model_out = self.model(x)
        out = model_out.mean(dim=(-2, -1), keepdim=True)
        return out


class TwoBranchDownsamplingBackbone(nn.Module):
    """
        Resnet backbone
        N, input_nc, H, W -> N, init_nc*(2**n_downsampling), 1, 1
        2 branch output
    """

    def __init__(self, input_nc, init_nc=64, n_downsampling=4, n_blocks=3, norm_type='instance', use_dropout=False):
        super(TwoBranchDownsamplingBackbone, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.projection = nn.Sequential(
            nn.Conv2d(input_nc, init_nc, kernel_size=7, stride=1, padding=3, bias=use_bias),
            norm_layer(init_nc),
            nn.LeakyReLU(0.1, inplace=True)
        )

        sequence1 = []
        sequence2 = []

        dim = init_nc
        for i in range(n_downsampling):  # downsample the feature map
            sequence1 += [
                nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(2 * dim),
                nn.LeakyReLU(0.1, inplace=True),
            ]
            sequence2 += [
                nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(2 * dim),
                nn.LeakyReLU(0.1, inplace=True),
            ]
            dim *= 2

        for i in range(n_blocks):  # ResBlock
            sequence1 += [
                ResnetBlock(dim, norm_layer, use_dropout, use_bias)
            ]
            sequence2 += [
                ResnetBlock(dim, norm_layer, use_dropout, use_bias)
            ]

        self.branch1 = nn.Sequential(*sequence1)
        self.branch2 = nn.Sequential(*sequence2)

    def forward(self, x):
        projection_out = self.projection(x)
        branch1_out = self.branch1(projection_out)
        branch2_out = self.branch2(projection_out)

        out1 = branch1_out.mean(dim=(-2, -1), keepdim=True)
        out2 = branch2_out.mean(dim=(-2, -1), keepdim=True)
        return out1, out2
