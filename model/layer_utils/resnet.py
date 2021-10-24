import functools

import torch
import torch.nn as nn

from .funcs import get_norm_layer
from .region_non_local_block import RegionNonLocalBlock


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
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            sequence += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = x + self.model(x)
        return out


class ResnetBackbone(nn.Module):
    """
        Resnet backbone
        input_nc -> output_nc
    """

    def __init__(self, input_nc, output_nc=64, n_downsampling=2, n_blocks=3, norm_type='instance', use_dropout=False):
        super(ResnetBackbone, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        sequence = [
            nn.Conv2d(input_nc, output_nc, kernel_size=7, stride=1, padding=3, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(True)
        ]

        dim = output_nc
        for i in range(n_downsampling):  # downsample the feature map
            sequence += [
                nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(2 * dim),
                nn.ReLU(True)
            ]
            dim *= 2

        for i in range(n_blocks):  # ResBlock
            sequence += [
                ResnetBlock(dim, norm_layer, use_dropout, use_bias)
            ]

        for i in range(n_downsampling):  # upsample the feature map
            sequence += [
                nn.ConvTranspose2d(dim, dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                norm_layer(dim // 2),
                nn.ReLU(True)
            ]
            dim //= 2

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x)
        return out


class NonLocalResnetDownsamplingBlock(nn.Module):
    """
        non-local Resnet downsampling block
        in_channel -> out_channel
    """

    def __init__(self, in_channel, out_channel, norm_layer, use_dropout, use_bias, latent_dim):
        super(NonLocalResnetDownsamplingBlock, self).__init__()

        self.projection = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
        self.non_local = RegionNonLocalBlock(out_channel, latent_dim)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
        )
        out_sequence = [
            norm_layer(out_channel),
            nn.ReLU(inplace=True)
        ]

        if use_dropout:
            out_sequence += [nn.Dropout(0.5)]
        out_sequence += [nn.MaxPool2d(2)]

        self.out_block = nn.Sequential(*out_sequence)

    def forward(self, x):
        x_ = self.projection(x)
        x_ = self.non_local(x_)
        out = self.out_block(x_ + self.bottleneck(x_))
        return out


class NonLocalResnetUpsamplingBlock(nn.Module):
    """
        non-local Resnet upsampling block
        x1:in_channel1  x2:in_channel2  -->  out_channel
    """

    def __init__(self, in_channel1, in_channel2, out_channel, norm_layer, use_dropout, use_bias, latent_dim):
        super(NonLocalResnetUpsamplingBlock, self).__init__()
        # in_channel1: 待上采样的输入通道数
        # in_channel2: skip link来的通道数
        self.upsample = nn.ConvTranspose2d(in_channel1, in_channel1 // 2, kernel_size=4, stride=2, padding=1,
                                           bias=use_bias)
        self.projection = nn.Conv2d(in_channel1 // 2 + in_channel2, out_channel, kernel_size=1, stride=1)
        self.non_local = RegionNonLocalBlock(out_channel, latent_dim)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
        )
        out_sequence = [
            norm_layer(out_channel),
            nn.ReLU(inplace=True)
        ]

        if use_dropout:
            out_sequence += [nn.Dropout(0.5)]

        self.out_block = nn.Sequential(*out_sequence)

    def forward(self, x1, x2):
        # x1: 待上采样的输入
        # x2: skip link来的输入
        x_ = self.projection(torch.cat([x2, self.upsample(x1)], dim=1))
        x_ = self.non_local(x_)
        out = self.out_block(x_ + self.bottleneck(x_))
        return out


class NonLocalResnetBackbone(nn.Module):
    """
        non-local Resnet backbone
        input_nc -> output_nc
    """

    def __init__(self, input_nc, output_nc=64, n_downsampling=2, n_blocks=6, norm_type='instance', use_dropout=False,
                 latent_dim=8):
        super(NonLocalResnetBackbone, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.n_downsampling = n_downsampling
        self.n_blocks = n_blocks

        self.projection = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=7, stride=1, padding=3, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(True)
        )
        self.in_conv = nn.Sequential(
            nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(2 * output_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nn.ReLU(True)
        )
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()

        dim = output_nc
        for i in range(n_downsampling):
            self.downsampling_blocks.append(
                NonLocalResnetDownsamplingBlock(dim, 2 * dim, norm_layer, use_dropout, use_bias, latent_dim)
            )
            dim *= 2

        res_blocks_seq = n_blocks * [ResnetBlock(dim, norm_layer, use_dropout, use_bias)]
        self.res_blocks = nn.Sequential(*res_blocks_seq)

        for i in range(n_downsampling):
            self.upsampling_blocks.append(
                NonLocalResnetUpsamplingBlock(dim, dim // 2, dim // 2, norm_layer, use_dropout, use_bias, latent_dim)
            )
            dim //= 2

    def forward(self, x):
        x_ = self.projection(x)
        out = self.in_conv(x_)

        skip_links = list()
        for i in range(self.n_downsampling):
            skip_links.append(out)
            out = self.downsampling_blocks[i](out)

        out = self.res_blocks(out)

        for i in range(self.n_downsampling):
            out = self.upsampling_blocks[i](out, skip_links[-i - 1])

        out = self.out_conv(torch.cat([x_, out], dim=1))
        return out
