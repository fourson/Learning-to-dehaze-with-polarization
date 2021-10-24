import functools

import torch.nn as nn

from .funcs import get_norm_layer


class AttentionBlock(nn.Module):
    """
        attention block
        x:in_channel_x  g:in_channel_g  -->  in_channel_x
    """

    def __init__(self, in_channel_x, in_channel_g, channel_t, norm_type='instance'):
        super(AttentionBlock, self).__init__()

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        # in_channel_x: input channels
        # in_channel_g: gating signal channels
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
        # x: (N, in_channel_x, H, W) input signal
        # g: (N, in_channel_g, H, W) gating signal
        # both x and g have the same H and W
        x_out = self.x_block(x)  # (N, channel_t, H, W)
        g_out = self.g_block(g)  # (N, channel_t, H, W)
        t_in = self.relu(x_out + g_out)  # (N, 1, H, W)
        attention_map = self.t_block(t_in)  # (N, 1, H, W)
        return x * attention_map  # (N, in_channel_x, H, W)
