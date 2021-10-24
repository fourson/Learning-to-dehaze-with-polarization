import torch
import torch.nn as nn


class TVLoss(nn.Module):
    """
        total variation (TV) loss encourages spatial smoothness
    """

    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        N, C, H, W = x.size()
        count_h = C * (H - 1) * W
        count_w = C * H * (W - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :H - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :W - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / N
