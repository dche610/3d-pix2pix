import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class GatedConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=2, padding=1, dilation=1,
        groups=1, bias=True, norm=None, activation=nn.LeakyReLU(0.2, inplace=True),
    ):
        super().__init__()
        self.gatingConv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias)
        self.featureConv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias)
        if norm == 'SN':
            self.gatingConv = nn.utils.spectral_norm(self.gatingConv)
            self.featureConv = nn.utils.spectral_norm(self.featureConv)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature = self.featureConv(x)
        gating = self.gatingConv(x)
        gated_mask = self.sigmoid(gating)
        x = feature * gated_mask
        return x
        

class GatedDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True),
                 transpose=False, output_padding=0):
        super().__init__()
        self.featureConv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride, padding, output_padding, groups, bias, dilation)
        self.gatingConv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride, padding, output_padding, groups, bias, dilation)
        if norm == 'SN':
            self.gatingConv = nn.utils.spectral_norm(self.gatingConv)
            self.featureConv = nn.utils.spectral_norm(self.featureConv)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature = self.featureConv(x)
        gating = self.gatingConv(x)
        gated_mask = self.sigmoid(gating)
        x = feature * gated_mask
        return x