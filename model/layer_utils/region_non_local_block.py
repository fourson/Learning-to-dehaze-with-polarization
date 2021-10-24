import torch
from torch import nn
from .non_local_block import Self_Attn_FM


class RegionNonLocalBlock(nn.Module):
    """
        region non-local block
        in_channel -> in_channel
    """

    def __init__(self, in_channel, latent_dim=2, subsample=True, grid=(8, 8)):
        super(RegionNonLocalBlock, self).__init__()

        self.non_local_block = Self_Attn_FM(in_channel, latent_dim=latent_dim, subsample=subsample)
        self.grid = grid

    def forward(self, x):
        input_row_list = x.chunk(self.grid[0], dim=2)
        output_row_list = []
        for i, row in enumerate(input_row_list):
            input_grid_list_of_a_row = row.chunk(self.grid[1], dim=3)
            output_grid_list_of_a_row = []
            for j, grid in enumerate(input_grid_list_of_a_row):
                grid = self.non_local_block(grid)
                output_grid_list_of_a_row.append(grid)
            output_row = torch.cat(output_grid_list_of_a_row, dim=3)
            output_row_list.append(output_row)
        output = torch.cat(output_row_list, dim=2)
        return output


class RegionNonLocalEnhancedDenseBlock(nn.Module):
    """
        region non-local enhanced dense block
        in_channel -> in_channel
    """

    def __init__(self, in_channel=64, inter_channel=32, n_blocks=3, latent_dim=2, subsample=True, grid=(8, 8)):
        super(RegionNonLocalEnhancedDenseBlock, self).__init__()

        self.region_non_local = RegionNonLocalBlock(in_channel, latent_dim, subsample, grid)
        self.conv_blocks = nn.ModuleList()

        dim = in_channel
        for i in range(n_blocks):
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=dim, out_channels=inter_channel, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                )
            )
            dim += inter_channel

        self.fusion = nn.Conv2d(in_channels=dim, out_channels=in_channel, kernel_size=1, stride=1)

    def forward(self, x):
        feature_list = [self.region_non_local(x)]
        for conv_block in self.conv_blocks:
            feature_list.append(conv_block(torch.cat(feature_list, dim=1)))
        out = self.fusion(torch.cat(feature_list, dim=1)) + x
        return out

