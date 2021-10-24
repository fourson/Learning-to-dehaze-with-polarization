import torch
import torch.nn as nn


class PerPixelPredictionBlock(nn.Module):
    """
        predict the fold_number per pixel
        (N, input_nc, H, W) -> (N, input_nc, n_label, H, W)
    """

    def __init__(self, input_nc, n_label):
        super(PerPixelPredictionBlock, self).__init__()

        # input_nc: input channel
        # n_label: label num
        self.conv_layers = nn.ModuleList(input_nc * [nn.Conv2d(1, n_label, kernel_size=1, stride=1)])
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # x: (N, C, H, W)
        t = list()
        for i in range(x.shape[1]):
            temp = x[:, i, :, :]  # (N, H, W)
            temp = torch.unsqueeze(temp, dim=1)  # (N, 1, H, W)
            temp = self.conv_layers[i](temp)  # (N, n_label, H, W)
            t.append(self.log_softmax(temp))

        # out = torch.cat(t, dim=1) # this will get an (N, C*n_label, H, W) tensor which has the shape we don't want
        out = torch.stack(t, dim=1)  # (N, C, n_label, H, W)

        return out
