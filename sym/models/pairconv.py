import torch
import torch.nn as nn
from sym.models.utils import (
    BasicBlock,
    ConvBnReLU,
    ConvBnReLU3D,
    ConvTrBnReLU3D,
    depth_softargmin,
    resample,
    warp,
)

class PairConv(nn.Module):
    def __init__(self, C_in, C_out):
        super(PairConv, self).__init__()
        self.pairconv = nn.Sequential(
            # BasicBlock(C_in, C_out, 1),  # C, H, W
            BasicBlock(C_in, C_in//2, 2),
            BasicBlock(C_in//2, C_in//4, 2),
            BasicBlock(C_in//4, C_out, 2),  # C, H/8, W/8
        )

    def forward(self, x_corr):
        B, N, C, H, W = x_corr.shape
        corr = self.pairconv(x_corr.view(B*N, C, H, W))  # [B, n, num_depth+C, H, W]
        # print('pairconv', corr.shape)
        corr = corr.view(B * N, -1).view(B, N, -1)
        return corr
