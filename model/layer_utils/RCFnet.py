import functools

import torch
import torch.nn as nn

from .funcs import get_norm_layer


class NonLinearRCFnetDoubleConvBlock(nn.Module):
    """
        RCFnet block
        in_channel  -->  o1:out_channel1  o2:out_channel2
    """

    def __init__(self, in_channel, out_channel1, out_channel2, norm_layer, use_bias):
        super(NonLinearRCFnetDoubleConvBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel1, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(out_channel1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel1, out_channel2, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(out_channel2)
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        return out1, out2


class NonLinearRCFnetBlock(nn.Module):
    """
        RCFnet block
        in_channel  -->  o1:out_channel1  o2:out_channel2
    """

    def __init__(self, in_channel, out_channel1, channel_t, out_channel2, n_branch, norm_layer, use_bias):
        super(NonLinearRCFnetBlock, self).__init__()

        self.branches = nn.ModuleList([
            NonLinearRCFnetDoubleConvBlock(in_channel, out_channel1, channel_t, norm_layer, use_bias)
        ])
        for i in range(n_branch - 1):
            self.branches.append(
                NonLinearRCFnetDoubleConvBlock(out_channel1, out_channel1, channel_t, norm_layer, use_bias)
            )
        self.out_conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_t, out_channel2, kernel_size=1, stride=1, bias=use_bias),
            norm_layer(out_channel2)
        )

    def forward(self, x):
        branch_outs = []
        t1 = x
        for branch in self.branches:
            t1, t2 = branch(t1)
            branch_outs.append(t2)
        out1 = t1
        out2 = self.out_conv(sum(branch_outs))
        return out1, out2


class NonLinearRFCnetBackbone(nn.Module):
    """
        RFC net backbone
        input_nc -> output_nc * (n_downsampling + 1)
    """

    def __init__(self, input_nc, output_nc=3, init_dim=64, mid_nc=21, n_downsampling=4, branch_num=(2, 2, 3, 3, 3),
                 norm_type='instance'):
        super(NonLinearRFCnetBackbone, self).__init__()

        assert n_downsampling + 1 == len(branch_num)
        self.n_downsampling = n_downsampling

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.pools = nn.ModuleList()
        self.blocks = nn.ModuleList([
            NonLinearRCFnetBlock(input_nc, init_dim, mid_nc, output_nc, branch_num[0], norm_layer, use_bias)
        ])
        self.deconvs = nn.ModuleList()
        self.out_conv = nn.Conv2d(output_nc * (n_downsampling + 1), output_nc, kernel_size=1, stride=1, bias=use_bias)

        dim = init_dim
        for i in range(n_downsampling):
            self.pools.append(nn.MaxPool2d(2))
            self.blocks.append(
                NonLinearRCFnetBlock(dim, 2 * dim, mid_nc, output_nc, branch_num[i + 1], norm_layer, use_bias)
            )
            dim *= 2
            self.deconvs.append(
                nn.ConvTranspose2d(output_nc, output_nc, kernel_size=2 ** (i + 2), stride=2 ** (i + 1), padding=2 ** i)
            )

    def forward(self, x):
        block_outs = []
        t1, t2 = self.blocks[0](x)
        block_outs.append(t2)
        for i in range(self.n_downsampling):
            t1 = self.pools[i](t1)
            t1, t2 = self.blocks[i + 1](t1)
            t2 = self.deconvs[i](t2)
            block_outs.append(t2)
        out = self.out_conv(torch.cat(block_outs, dim=1))
        return out
