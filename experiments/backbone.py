import torch.nn as nn
from .backbone_resnet import *
from .backbone_unet_encoder import unet_encoder
import torch
import time
from torch import nn
import math

# class eca_layer(nn.Module):
#     def __init__(self, channel, k_size):
#         super(eca_layer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool3d(1)
#         self.k_size = k_size
#         self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, bias=False, groups=channel)
#         self.sigmoid = nn.Sigmoid()
#
#
#
#     def forward(self, x):
#         # x: input features with shape [b, c, h, w]
#         b, c, z, h, w = x.size()
#
#         # feature descriptor on the global spatial information
#         y = self.avg_pool(x)
#
#         # Two different branches of ECA module
#         y = self.conv(y.squeeze(-1).squeeze(-2).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-2).unsqueeze(-1)
#
#         # Multi-scale information fusion
#         y = self.sigmoid(y)
#
#         return x * y.expand_as(x)

class EcaLayer(nn.Module):

    def __init__(self, gamma=2, b=1):
        super(EcaLayer, self).__init__()
        self.gamma = gamma
        self.b = b
        self.avg_pool = nn.AdaptiveAvgPool3d(1)



        # self.k_size = k_size
        # self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, bias=False, groups=channel)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, z, h, w = x.size()

        t = int(abs((math.log(c, 2) + self.b) / self.gamma))
        k_size = t if t % 2 else t + 1

        conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False).cuda()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = conv(y.squeeze(-1).squeeze(-2).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

__all__ = ['Backbone']

class Backbone(nn.Module):
    def __init__(self, backbone):
        super(Backbone, self).__init__()
        #self.nclass = nclass

        self.eca = EcaLayer()

        if backbone == 'resnet3d18':
            self.pretrained = resnet3d18()
        elif backbone == 'resnet3d34':
            self.pretrained = resnet3d34()
        elif backbone == 'resnet3d50':
            self.pretrained = resnet3d50()
        elif backbone == 'unet_encoder':
            self.pretrained = unet_encoder()
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        #upsample options
        self._up_kwargs = up_kwargs

    def backbone_forward(self, x):
        # x = self.eca(x)
        # # print("输入:",x.shape)
        # conv1 = self.pretrained.dconv_down1(x)
        # # print("conv1:", conv1.shape)
        # x1 = self.pretrained.maxpool(conv1)
        # # print("x1:",x1.shape)
        #
        # conv2 = self.pretrained.dconv_down2(x1)
        # # print("conv2:", conv2.shape)
        # x2 = self.pretrained.maxpool(conv2)
        # # print("x2:", x2.shape)
        #
        # conv3 = self.pretrained.dconv_down3(x2)
        # # print("conv3:", conv3.shape)
        # x3 = self.pretrained.maxpool(conv3)
        # # print("x3:", x3.shape)
        #
        # conv4 = self.pretrained.dconv_down4(x3)
        # # print("conv4:", conv4.shape)

        # x = self.eca(x)
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        x = self.eca(x)
        conv1 = self.pretrained.layer1(x)
        # print("conv1:",conv1.shape)

        x = self.eca(conv1)
        conv2 = self.pretrained.layer2(x)
        # print("conv2:",conv2.shape)

        x = self.eca(conv2)
        conv3 = self.pretrained.layer3(x)
        # print("conv3:",conv3.shape)

        x = self.eca(conv3)
        conv4 = self.pretrained.layer4(x)
        # print("conv4:",conv4.shape)

        return conv1, conv2, conv3, conv4
