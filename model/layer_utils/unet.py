import functools

import torch
import torch.nn as nn

from .funcs import get_norm_layer


class UnetDoubleConvBlock(nn.Module):
    """
        Unet double Conv block
        in_channel -> out_channel
    """

    def __init__(self, in_channel, out_channel, norm_layer, use_dropout, use_bias, mode='default'):
        super(UnetDoubleConvBlock, self).__init__()

        self.mode = mode

        if self.mode == 'default':
            self.model = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True)
            )
            out_sequence = []
        elif self.mode == 'bottleneck':
            self.model = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True)
            )
            out_sequence = []
        elif self.mode == 'res-bottleneck':
            self.projection = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
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
        else:
            raise NotImplementedError('mode [%s] is not found' % self.mode)

        if use_dropout:
            out_sequence += [nn.Dropout(0.5)]

        self.out_block = nn.Sequential(*out_sequence)

    def forward(self, x):
        if self.mode == 'res-bottleneck':
            x_ = self.projection(x)
            out = self.out_block(x_ + self.bottleneck(x_))
        else:
            out = self.out_block(self.model(x))
        return out


class UnetDownsamplingBlock(nn.Module):
    """
        Unet downsampling block
        in_channel -> out_channel
    """

    def __init__(self, in_channel, out_channel, norm_layer, use_dropout, use_bias, use_conv, mode='default'):
        super(UnetDownsamplingBlock, self).__init__()

        downsampling_layers = list()
        if use_conv:
            downsampling_layers += [
                nn.Conv2d(in_channel, in_channel, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(out_channel),
                nn.ReLU(inplace=True)
            ]
        else:
            downsampling_layers += [nn.MaxPool2d(2)]

        self.model = nn.Sequential(
            nn.Sequential(*downsampling_layers),
            UnetDoubleConvBlock(in_channel, out_channel, norm_layer, use_dropout, use_bias, mode=mode)
        )

    def forward(self, x):
        out = self.model(x)
        return out


class UnetUpsamplingBlock(nn.Module):
    """
        Unet upsampling block
        x1:in_channel1  x2:in_channel2  -->  out_channel
    """

    def __init__(self, in_channel1, in_channel2, out_channel, norm_layer, use_dropout, use_bias, mode='default'):
        super(UnetUpsamplingBlock, self).__init__()
        # in_channel1: 待上采样的输入通道数
        # in_channel2: skip link来的通道数
        self.upsample = nn.ConvTranspose2d(in_channel1, in_channel1 // 2, kernel_size=4, stride=2, padding=1,
                                           bias=use_bias)
        self.double_conv = UnetDoubleConvBlock(in_channel1 // 2 + in_channel2, out_channel, norm_layer, use_dropout,
                                               use_bias, mode=mode)

    def forward(self, x1, x2):
        # x1: 待上采样的输入
        # x2: skip link来的输入
        out = torch.cat([x2, self.upsample(x1)], dim=1)
        out = self.double_conv(out)
        return out


class UnetBackbone(nn.Module):
    """
        Unet backbone
        input_nc -> output_nc
    """

    def __init__(self, input_nc, output_nc=64, n_downsampling=4, use_conv_to_downsample=True, norm_type='instance',
                 use_dropout=False, mode='default'):
        super(UnetBackbone, self).__init__()

        self.n_downsampling = n_downsampling

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.double_conv_block = UnetDoubleConvBlock(input_nc, output_nc, norm_layer, use_dropout, use_bias, mode=mode)
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()

        dim = output_nc
        for i in range(n_downsampling):
            self.downsampling_blocks.append(
                UnetDownsamplingBlock(dim, 2 * dim, norm_layer, use_dropout, use_bias, use_conv_to_downsample,
                                      mode=mode)
            )
            dim *= 2

        for i in range(n_downsampling):
            self.upsampling_blocks.append(
                UnetUpsamplingBlock(dim, dim // 2, dim // 2, norm_layer, use_dropout, use_bias, mode=mode)
            )
            dim //= 2

    def forward(self, x):
        double_conv_block_out = self.double_conv_block(x)

        downsampling_blocks_out = list()
        downsampling_blocks_out.append(
            self.downsampling_blocks[0](double_conv_block_out)
        )
        for i in range(1, self.n_downsampling):
            downsampling_blocks_out.append(
                self.downsampling_blocks[i](downsampling_blocks_out[-1])
            )

        upsampling_blocks_out = list()
        upsampling_blocks_out.append(
            self.upsampling_blocks[0](downsampling_blocks_out[-1], downsampling_blocks_out[-2])
        )
        for i in range(1, self.n_downsampling - 1):
            upsampling_blocks_out.append(
                self.upsampling_blocks[i](upsampling_blocks_out[-1], downsampling_blocks_out[-2 - i])
            )
        upsampling_blocks_out.append(
            self.upsampling_blocks[-1](upsampling_blocks_out[-1], double_conv_block_out)
        )

        out = upsampling_blocks_out[-1]
        return out


class AttentionBlock(nn.Module):
    """
        attention block
        x:in_channel_x  g:in_channel_g  -->  in_channel_x
    """

    def __init__(self, in_channel_x, in_channel_g, channel_t, norm_layer, use_bias):
        # in_channel_x: 输入通道数(skip link来的)
        # in_channel_g: gating signal的通道数(上采样后的)
        super(AttentionBlock, self).__init__()
        self.x_block = nn.Sequential(
            nn.Conv2d(in_channel_x, channel_t, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(channel_t)
        )

        self.g_block = nn.Sequential(
            nn.Conv2d(in_channel_g, channel_t, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(channel_t)
        )

        self.t_block = nn.Sequential(
            nn.Conv2d(channel_t, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # x: (N, in_channel_x, H, W) 输入(skip link来的)
        # g: (N, in_channel_g, H, W) gating signal的输入(上采样后的)
        # x g两者的H W是一致的
        x_out = self.x_block(x)  # (N, channel_t, H, W)
        g_out = self.g_block(g)  # (N, channel_t, H, W)
        t_in = self.relu(x_out + g_out)  # (N, 1, H, W)
        attention_map = self.t_block(t_in)  # (N, 1, H, W)
        return x * attention_map  # (N, in_channel_x, H, W)


class AttentionUnetUpsamplingBlock(nn.Module):
    """
        attention Unet upsampling block
        x1:in_channel1  x2:in_channel2  -->  out_channel
    """

    def __init__(self, in_channel1, in_channel2, out_channel, norm_layer, use_dropout, use_bias, mode='default'):
        super(AttentionUnetUpsamplingBlock, self).__init__()
        # in_channel1: 待上采样的输入通道数
        # in_channel2: skip link来的通道数
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channel1, in_channel1 // 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(out_channel),
            nn.ReLU(inplace=True)
        )
        self.attention = AttentionBlock(in_channel1 // 2, in_channel2, in_channel1 // 2, norm_layer, use_bias)
        self.double_conv = UnetDoubleConvBlock(in_channel1 // 2 + in_channel2, out_channel, norm_layer, use_dropout,
                                               use_bias, mode=mode)

    def forward(self, x1, x2):
        # x1: 待上采样的输入
        # x2: skip link来的输入
        upsampled_x1 = self.upsample(x1)  # 作为attention block的gating signal
        attentioned_x2 = self.attention(x2, upsampled_x1)
        out = torch.cat([attentioned_x2, upsampled_x1], dim=1)
        out = self.double_conv(out)
        return out


class AttentionUnetBackbone(nn.Module):
    """
        attention Unet backbone
        input_nc -> output_nc
    """

    def __init__(self, input_nc, output_nc=64, n_downsampling=4, use_conv_to_downsample=False, norm_type='instance',
                 use_dropout=False, mode='default'):
        super(AttentionUnetBackbone, self).__init__()

        self.n_downsampling = n_downsampling

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.double_conv_block = UnetDoubleConvBlock(input_nc, output_nc, norm_layer, use_dropout, use_bias, mode=mode)
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()

        dim = output_nc
        for i in range(n_downsampling):
            self.downsampling_blocks.append(
                UnetDownsamplingBlock(dim, 2 * dim, norm_layer, use_dropout, use_bias, use_conv_to_downsample,
                                      mode=mode)
            )
            dim *= 2

        for i in range(n_downsampling):
            self.upsampling_blocks.append(
                AttentionUnetUpsamplingBlock(dim, dim // 2, dim // 2, norm_layer, use_dropout, use_bias, mode=mode)
            )
            dim //= 2

    def forward(self, x):
        double_conv_block_out = self.double_conv_block(x)

        downsampling_blocks_out = list()
        downsampling_blocks_out.append(
            self.downsampling_blocks[0](double_conv_block_out)
        )
        for i in range(1, self.n_downsampling):
            downsampling_blocks_out.append(
                self.downsampling_blocks[i](downsampling_blocks_out[-1])
            )

        upsampling_blocks_out = list()
        upsampling_blocks_out.append(
            self.upsampling_blocks[0](downsampling_blocks_out[-1], downsampling_blocks_out[-2])
        )
        for i in range(1, self.n_downsampling - 1):
            upsampling_blocks_out.append(
                self.upsampling_blocks[i](upsampling_blocks_out[-1], downsampling_blocks_out[-2 - i])
            )
        upsampling_blocks_out.append(
            self.upsampling_blocks[-1](upsampling_blocks_out[-1], double_conv_block_out)
        )

        out = upsampling_blocks_out[-1]
        return out
