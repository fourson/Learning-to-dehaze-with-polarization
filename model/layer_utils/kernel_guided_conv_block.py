import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelGuidedConvBlockCell(nn.Module):
    """
        kernel guided conv block cell
        kernel:(1, kernel_h, kernel_w)  x:(in_channel, h, w)  -->  (in_channel, h, w)
    """

    def __init__(self, kernel_h, kernel_w, in_channel):
        super(KernelGuidedConvBlockCell, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(kernel_h * kernel_w, in_channel),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            nn.ReLU(inplace=True)
        )
        self.linear3 = nn.Linear(in_channel, in_channel)
        self.linear4 = nn.Linear(in_channel, in_channel)
        self.learnable_bias = nn.Parameter(torch.ones(in_channel))
        self.feature_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, kernel, x):
        linear1_out = self.linear1(kernel.flatten(start_dim=1))
        linear2_out = self.linear2(linear1_out)
        multiplier = (self.linear3(linear2_out) + self.learnable_bias).unsqueeze(-1).unsqueeze(-1)
        bias = self.linear4(linear2_out).unsqueeze(-1).unsqueeze(-1)
        feature_conv_out = self.feature_conv(x)
        out = F.relu(feature_conv_out * multiplier + bias)
        return out


class KernelGuidedConvBlock(nn.Module):
    """
        kernel guided conv block
        kernel:(1, kernel_h, kernel_w)  x:(in_channel, h, w)  -->  (in_channel, h, w)
    """

    def __init__(self, kernel_h, kernel_w, in_channel, learn_residual=False):
        super(KernelGuidedConvBlock, self).__init__()

        self.cell1 = KernelGuidedConvBlockCell(kernel_h, kernel_w, in_channel)
        self.cell2 = KernelGuidedConvBlockCell(kernel_h, kernel_w, in_channel)
        self.cell3 = KernelGuidedConvBlockCell(kernel_h, kernel_w, in_channel)

        self.learn_residual = learn_residual

    def forward(self, kernel, x):
        cell1_out = self.cell1(kernel, x)
        cell2_out = self.cell2(kernel, cell1_out)
        cell3_out = self.cell3(kernel, cell2_out)
        if self.learn_residual:
            out = cell3_out + x
        else:
            out = cell3_out
        return out
