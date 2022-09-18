import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


########################
# Convolutional Blocks #
########################

class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True),
                 transpose=False, output_padding=0):
        super().__init__()
        if padding == -1:
            padding = ((np.array(kernel_size) - 1) * np.array(dilation)) // 2
            # to check if padding is not a 0-d array, otherwise tuple(padding) will raise an exception
            if hasattr(padding, '__iter__'):
                padding = tuple(padding)

        if transpose:
            self.conv = nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size,
                stride, padding, output_padding, groups, bias, dilation)
        else:
            self.conv = nn.Conv3d(
                in_channels, out_channels, kernel_size,
                stride, padding, dilation, groups, bias)

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm3d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm3d(out_channels, track_running_stats=True)
        elif norm == "SN":
            self.norm = None
            self.conv = nn.utils.spectral_norm(self.conv)
        elif norm is None:
            self.norm = None
        else:
            raise NotImplementedError(f"Norm type {norm} not implemented")

        self.activation = activation

    def forward(self, xs):
        out = self.conv(xs)
        if self.activation is not None:
            out = self.activation(out)
        if self.norm is not None:
            out = self.norm_layer(out)
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True),
                 transpose=False, output_padding=0):
        super().__init__()
        if padding == -1:
            padding = ((np.array(kernel_size) - 1) * np.array(dilation)) // 2
            if hasattr(padding, '__iter__'):
                padding = tuple(padding)

        if transpose:
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size,
                stride, padding, output_padding, groups, bias, dilation)
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride, padding, dilation, groups, bias)

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)
        elif norm == "SN":
            self.norm = None
            self.conv = nn.utils.spectral_norm(self.conv)
        elif norm is None:
            self.norm = None
        else:
            raise NotImplementedError(f"Norm type {norm} not implemented")

        self.activation = activation

    def forward(self, xs):
        out = self.conv(xs)
        if self.activation is not None:
            out = self.activation(out)
        if self.norm is not None:
            out = self.norm_layer(out)
        return out


class VanillaConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True), conv_by='3d'
    ):

        super().__init__()
        if conv_by == '3d':
            self.module = torch.nn
        else:
            raise NotImplementedError(f'conv_by {conv_by} is not implemented.')

        self.padding = tuple(((np.array(kernel_size) - 1) * np.array(dilation)) // 2) if padding == -1 else padding
        self.featureConv = self.module.Conv3d(
            in_channels, out_channels, kernel_size,
            stride, self.padding, dilation, groups, bias)

        self.norm = norm
        if norm == "BN":
            self.norm_layer = self.module.BatchNorm3d(out_channels)
        elif norm == "IN":
            self.norm_layer = self.module.InstanceNorm3d(out_channels, track_running_stats=True)
        elif norm == "SN":
            self.norm = None
            self.featureConv = nn.utils.spectral_norm(self.featureConv)
        elif norm is None:
            self.norm = None
        else:
            raise NotImplementedError(f"Norm type {norm} not implemented")

        self.activation = activation

    def forward(self, xs):
        out = self.featureConv(xs)
        if self.activation:
            out = self.activation(out)
        if self.norm is not None:
            out = self.norm_layer(out)
        return out


class VanillaDeconv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True),
        scale_factor=2, conv_by='3d'
    ):
        super().__init__()
        self.conv = VanillaConv(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, norm, activation, conv_by=conv_by)
        self.scale_factor = scale_factor

    def forward(self, xs):
        xs_resized = F.interpolate(xs, scale_factor=(1, self.scale_factor, self.scale_factor))
        return self.conv(xs_resized)


class GatedConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        groups=1, bias=True, norm=None
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
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        groups=1, bias=True, norm=None
    ):
        super().__init__()
        self.gatingConv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias)
        self.featureConv = nn.ConvTranspose3d(
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
        

# modified from https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/net.py
class PartialConv(VanillaConv):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True), conv_by='3d'):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, norm, activation, conv_by
        )
        self.mask_sum_conv = self.module.Conv3d(1, 1, kernel_size,
                                                stride, padding, dilation, groups, False)
        torch.nn.init.constant_(self.mask_sum_conv.weight, 1.0)

        # mask conv needs not update
        for param in self.mask_sum_conv.parameters():
            param.requires_grad = False

        if norm == "SN":
            self.featureConv = nn.utils.spectral_norm(self.featureConv)
            raise NotImplementedError(f"Norm type {norm} not implemented")

    def forward(self, input_tuple):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # output = W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0), if sum(M) != 0
        #        = 0, if sum(M) == 0
        inp, mask = input_tuple

        # C(M .* X)
        output = self.featureConv(mask * inp)

        # C(0) = b
        if self.featureConv.bias is not None:
            output_bias = self.featureConv.bias.view(1, -1, 1, 1, 1)
        else:
            output_bias = torch.zeros([1, 1, 1, 1, 1]).to(inp.device)

        # D(M) = sum(M)
        with torch.no_grad():
            mask_sum = self.mask_sum_conv(mask)

        # find those sum(M) == 0
        no_update_holes = (mask_sum == 0)

        # Just to prevent devided by 0
        mask_sum_no_zero = mask_sum.masked_fill_(no_update_holes, 1.0)

        # output = [C(M .* X) – C(0)] / D(M) + C(0), if sum(M) != 0
        #        = 0, if sum (M) == 0
        output = (output - output_bias) / mask_sum_no_zero + output_bias
        output = output.masked_fill_(no_update_holes, 0.0)

        # create a new mask with only 1 or 0
        new_mask = torch.ones_like(mask_sum)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        if self.activation is not None:
            output = self.activation(output)
        if self.norm is not None:
            output = self.norm_layer(output)
        return output, new_mask


class PartialDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True),
                 scale_factor=2, conv_by='3d'):
        super().__init__()
        self.conv = PartialConv(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, norm, activation, conv_by=conv_by)
        self.scale_factor = scale_factor

    def forward(self, input_tuple):
        inp, mask = input_tuple
        inp_resized = F.interpolate(inp, scale_factor=(1, self.scale_factor, self.scale_factor))
        with torch.no_grad():
            mask_resized = F.interpolate(mask, scale_factor=(1, self.scale_factor, self.scale_factor))
        return self.conv((inp_resized, mask_resized))