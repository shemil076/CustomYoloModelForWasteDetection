# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "ASFF",
    "DCNv2"
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=16):
        """
        Initialize the Channel Attention module.

        Args:
            in_planes (int): Number of input channels.
            ratio (int): Reduction ratio for the hidden channels in the channel attention block.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the Channel Attention module.

        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            out (torch.Tensor): Output tensor after applying channel attention.
        """

        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        """
        Initialize the Spatial Attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the Spatial Attention module.

        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            out (torch.Tensor): Output tensor after applying spatial attention.
        """

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    # ch_in, ch_out, shortcut, groups, expansion, ratio, kernel_size
    def __init__(self, c1, c2, kernel_size=3, shortcut=True, g=1, e=0.5, ratio=8):
        """
        Initialize the CBAM (Convolutional Block Attention Module)

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            shortcut (bool): Whether to use a shortcut connection.
            g (int): Number of groups for grouped convolutions.
            e (float): Expansion factor for hidden channels.
            ratio (int): Reduction ratio for the hidden channels in the channel attention block.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        self.channel_attention = ChannelAttention(c2, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Forward pass of the CBAM .

        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            out (torch.Tensor): Output tensor after applying the CBAM bottleneck.
        """

        x2 = self.cv2(self.cv1(x))
        out = self.channel_attention(x2) * x2
        out = self.spatial_attention(out) * out
        return x + out if self.add else out

# class ASFF(nn.Module):
#     def __init__(self, level, multiplier=1, rfb=False, vis=False, act_cfg=True):
#         """
#         multiplier should be 1, 0.5
#         which means, the channel of ASFF can be
#         512, 256, 128 -> multiplier=0.5
#         1024, 512, 256 -> multiplier=1
#         For even smaller, you need change code manually.
#         """
#         super(ASFF, self).__init__()
#         print("self", )
#         self.level = level
#         print("self.level", self.level)
#         print("multiplier",multiplier)
#         self.dim = [int(1024 * multiplier), int(512 * multiplier),
#                     int(256 * multiplier)]
#         print("self.dim",self.dim)

#         self.inter_dim = self.dim[self.level]
#         print("expected",  self.inter_dim)
#         print("Actual", self.dim[self.level])

#         if level == 0:
#             self.stride_level_1 = Conv(int(512 * multiplier), self.inter_dim, 3, 2)

#             self.stride_level_2 = Conv(int(256 * multiplier), self.inter_dim, 3, 2)

#             self.expand = Conv(self.inter_dim, int(1024 * multiplier), 3, 1)
#         elif level == 1:
#             self.compress_level_0 = Conv(int(1024 * multiplier), self.inter_dim, 1, 1)
#             self.stride_level_2 = Conv(int(256 * multiplier), self.inter_dim, 3, 2)
#             self.expand = Conv(self.inter_dim, int(512 * multiplier), 3, 1)
#         elif level == 2:
#             self.compress_level_0 = Conv(int(1024 * multiplier), self.inter_dim, 1, 1)
#             self.compress_level_1 = Conv(int(512 * multiplier), self.inter_dim, 1, 1)
#             self.expand = Conv(self.inter_dim, int(256 * multiplier), 3, 1)

#         # when adding rfb, we use half number of channels to save memory
#         compress_c = 8 if rfb else 16
#         self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
#         self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
#         self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

#         self.weight_levels = Conv(compress_c * 3, 3, 1, 1)
#         self.vis = vis

#     def forward(self, x):  # l,m,s
#         """
#         #
#         256, 512, 1024
#         from small -> large
#         """
#         # max feature
#         global level_0_resized, level_1_resized, level_2_resized
#         x_level_0 = x[2]
#         # mid feature
#         x_level_1 = x[1]
#         # min feature
#         x_level_2 = x[0]

#         if self.level == 0:
#             level_0_resized = x_level_0
#             level_1_resized = self.stride_level_1(x_level_1)
#             level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
#             level_2_resized = self.stride_level_2(level_2_downsampled_inter)
#         elif self.level == 1:
#             level_0_compressed = self.compress_level_0(x_level_0)
#             level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
#             level_1_resized = x_level_1
#             level_2_resized = self.stride_level_2(x_level_2)
#         elif self.level == 2:
#             level_0_compressed = self.compress_level_0(x_level_0)
#             level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
#             x_level_1_compressed = self.compress_level_1(x_level_1)
#             level_1_resized = F.interpolate(x_level_1_compressed, scale_factor=2, mode='nearest')
#             level_2_resized = x_level_2

#         level_0_weight_v = self.weight_level_0(level_0_resized)
#         level_1_weight_v = self.weight_level_1(level_1_resized)
#         level_2_weight_v = self.weight_level_2(level_2_resized)

#         levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
#         levels_weight = self.weight_levels(levels_weight_v)
#         levels_weight = F.softmax(levels_weight, dim=1)

#         fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
#                             level_1_resized * levels_weight[:, 1:2, :, :] + \
#                             level_2_resized * levels_weight[:, 2:, :, :]

#         out = self.expand(fused_out_reduced)

#         if self.vis:
#             return out, levels_weight, fused_out_reduced.sum(dim=1)
#         else:
#             return out
    

# def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
#     """
#     Add a conv2d / batchnorm / leaky ReLU block.
#     Args:
#         in_ch (int): number of input channels of the convolution layer.
#         out_ch (int): number of output channels of the convolution layer.
#         ksize (int): kernel size of the convolution layer.
#         stride (int): stride of the convolution layer.
#     Returns:
#         stage (Sequential) : Sequential layers composing a convolution block.
#     """
#     stage = nn.Sequential()
#     pad = (ksize - 1) // 2
#     stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
#                                        out_channels=out_ch, kernel_size=ksize, stride=stride,
#                                        padding=pad, bias=False))
#     stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
#     if leaky:
#         stage.add_module('leaky', nn.LeakyReLU(0.1))
#     else:
#         stage.add_module('relu6', nn.ReLU6(inplace=True))
#     return stage

# class ASFF(nn.Module):
#     def __init__(self, level, rfb=False, vis=False):
#         super(ASFF, self).__init__()
#         print("Running ASFF")
#         print("Level is", level)
#         self.level = level
#         self.dim = [512, 256, 256]
#         self.inter_dim = self.dim[self.level]
#         if level==0:
#             self.stride_level_1 = add_conv(256, self.inter_dim, 3, 2)
#             self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2)
#             self.expand = add_conv(self.inter_dim, 1024, 3, 1)
#         elif level==1:
#             self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
#             self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2)
#             self.expand = add_conv(self.inter_dim, 512, 3, 1)
#         elif level==2:
#             self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
#             self.expand = add_conv(self.inter_dim, 256, 3, 1)

#         compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory

#         self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
#         self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
#         self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

#         self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)
#         self.vis= vis


#     def forward(self, x_level_0, x_level_1, x_level_2):
#         if self.level==0:
#             level_0_resized = x_level_0
#             level_1_resized = self.stride_level_1(x_level_1)

#             level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1)
#             level_2_resized = self.stride_level_2(level_2_downsampled_inter)

#         elif self.level==1:
#             level_0_compressed = self.compress_level_0(x_level_0)
#             level_0_resized =F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
#             level_1_resized =x_level_1
#             level_2_resized =self.stride_level_2(x_level_2)
#         elif self.level==2:
#             level_0_compressed = self.compress_level_0(x_level_0)
#             level_0_resized =F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
#             level_1_resized =F.interpolate(x_level_1, scale_factor=2, mode='nearest')
#             level_2_resized =x_level_2

#         level_0_weight_v = self.weight_level_0(level_0_resized)
#         level_1_weight_v = self.weight_level_1(level_1_resized)
#         level_2_weight_v = self.weight_level_2(level_2_resized)
#         levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v),1)
#         levels_weight = self.weight_levels(levels_weight_v)
#         levels_weight = F.softmax(levels_weight, dim=1)

#         fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
#                             level_1_resized * levels_weight[:,1:2,:,:]+\
#                             level_2_resized * levels_weight[:,2:,:,:]

#         out = self.expand(fused_out_reduced)

#         if self.vis:
#             return out, levels_weight, fused_out_reduced.sum(dim=1)
#         else:
#             return out    

class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)

class DCNv2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        groups=1,
        dilation=1,
        act=True,
        deformable_groups=1,
    ):
        super(DCNv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        padding = autopad(kernel_size, padding, dilation)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        )
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = (
            Conv.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1],
            self.groups,
            self.deformable_groups,
            True,
        )
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()