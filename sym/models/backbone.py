import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sym.config import CI, CM
from sym.models.utils import (
    BasicBlock,
    ConvBnReLU,
    ConvBnReLU3D,
    ConvTrBnReLU3D,
    depth_softargmin,
    resample,
    warp,
)
from sym.utils import benchmark


class FeatureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(4, 64, 5, 2, 2),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64, 2),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
        )

    def forward(self, x):
        return self.backbone(x)

