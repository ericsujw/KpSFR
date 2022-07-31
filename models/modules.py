"""
Modifed from STCN: https://github.com/hkchengrex/STCN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from models import mod_resnet


class ResBlock(nn.Module):

    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()

        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(
                indim, outdim, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class FeatureFusionBlock(nn.Module):

    def __init__(self, indim, outdim):
        super().__init__()

        self.block1 = ResBlock(indim, outdim)
        self.block2 = ResBlock(outdim, outdim)

    def forward(self, x, f32):

        x = torch.cat([x, f32], dim=1)
        x = self.block1(x)
        x = self.block2(x)

        return x


# Multiple objects version, used in other times
class ValueEncoder(nn.Module):

    def __init__(self, num_objects, non_local):
        super().__init__()

        resnet = mod_resnet.resnet18(
            pretrained=True, extra_chan=1, non_local=non_local)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 1/4, 64
        self.layer2 = resnet.layer2  # 1/8, 128
        self.layer3 = resnet.layer3  # 1/16, 256
        self.layer4 = resnet.layer4  # 1/32, 512

        self.fuser = FeatureFusionBlock(512 + 512, 256)

    def forward(self, image, key_f32, mask, other_masks, cls, isFirst):
        # input mask: b*1*h*w
        # key_f32 is the feature from the key encoder
        assert mask.shape[1] == 1, 'channel inconsistent'
        if isFirst:
            mask = F.interpolate(mask, scale_factor=4, mode='nearest')

        assert image.shape[-2:] == mask.shape[-2:], 'shape inconsistent'
        f = torch.cat([image, mask], dim=1)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64

        x = self.layer1(x)   # 1/4, 64
        x = self.layer2(x)  # 1/8, 128
        x = self.layer3(x)  # 1/16, 256
        x = self.layer4(x)  # 1/32, 512

        x = self.fuser(x, key_f32)

        return x


class KeyEncoder(nn.Module):

    def __init__(self, num_objects, non_local):
        super().__init__()

        resnet = mod_resnet.resnet34(pretrained=True, non_local=non_local)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 1/4, 64
        self.layer2 = resnet.layer2  # 1/8, 128
        self.layer3 = resnet.layer3  # 1/16, 256
        self.layer4 = resnet.layer4  # 1/32, 512

    def forward(self, f, cls):

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64

        f4 = self.layer1(x)   # 1/4, 64
        f8 = self.layer2(f4)  # 1/8, 128
        f16 = self.layer3(f8)  # 1/16, 256
        f32 = self.layer4(f16)  # 1/32, 512

        return f32, f16, f8, f4


class KeyProjection(nn.Module):

    def __init__(self, indim, keydim):
        super().__init__()

        self.key_proj = nn.Conv2d(
            indim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        if self.key_proj.bias is not None:
            nn.init.zeros_(self.key_proj.bias.data)

    def forward(self, x):

        return self.key_proj(x)
