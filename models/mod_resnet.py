"""
Modifed from STCN: https://github.com/hkchengrex/STCN
mod_resnet.py - A modified ResNet structure
We append extra channels to the first conv by some network surgery
"""

from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.utils import model_zoo

from models.non_local import NLBlockND


def weights_init(m):

    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def weights_init2(m):

    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()


def load_weights_sequential(target, source_state, extra_chan=1):

    new_dict = OrderedDict()

    for k1, v1 in target.state_dict().items():
        if not 'num_batches_tracked' in k1:
            if k1 in source_state:
                tar_v = source_state[k1]

                if v1.shape != tar_v.shape:
                    # Init the new segmentation channel with zeros
                    c, _, w, h = v1.shape
                    pads = torch.zeros((c, extra_chan, w, h),
                                       device=tar_v.device)
                    nn.init.orthogonal_(pads)
                    tar_v = torch.cat([tar_v, pads], 1)

                new_dict[k1] = tar_v

    target.load_state_dict(new_dict, strict=False)


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(
            inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers=(2, 2, 2, 2), extra_chan=1, non_local=False, sub_sample=False, bn_layer=False):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(
            3 + extra_chan, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        # TODO: add a few dilated convolution and spatial-only non-local block layers
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilation=2, non_local=non_local, sub_sample=sub_sample, bn_layer=bn_layer)
        self.layer4[0].apply(weights_init)
        self.layer4[2].apply(weights_init)
        self.layer4[1].apply(weights_init2)
        self.layer4[3].apply(weights_init2)

    def _make_layer(self, block, planes, num_blocks, stride=1, dilation=1, non_local=False, sub_sample=False, bn_layer=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        last_idx = len(strides)

        if non_local:
            last_idx = len(strides) - 1

        for i in range(last_idx):
            layers.append(
                block(self.inplanes, planes, strides[i], downsample, dilation))
            self.inplanes = planes * block.expansion
            downsample = None

        if non_local:
            layers.append(NLBlockND(in_channels=planes, mode='embedded',
                                    dimension=2, sub_sample=sub_sample, bn_layer=bn_layer))
            layers.append(
                block(self.inplanes, planes, strides[-1], dilation=dilation))
            layers.append(NLBlockND(in_channels=planes, mode='embedded',
                                    dimension=2, sub_sample=sub_sample, bn_layer=bn_layer))

        return nn.Sequential(*layers)


class ResNet34(nn.Module):

    def __init__(self, block, layers=(2, 2, 2, 2), non_local=False, sub_sample=False, bn_layer=False):
        self.inplanes = 64
        super(ResNet34, self).__init__()

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        # TODO: add a few dilated convolution and spatial-only non-local block layers
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilation=2, non_local=non_local, sub_sample=sub_sample, bn_layer=bn_layer)
        self.layer4[0].apply(weights_init)
        self.layer4[1].apply(weights_init)
        self.layer4[3].apply(weights_init)
        self.layer4[2].apply(weights_init2)
        self.layer4[4].apply(weights_init2)

    def _make_layer(self, block, planes, num_blocks, stride=1, dilation=1, non_local=False, sub_sample=False, bn_layer=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        last_idx = len(strides)

        if non_local:
            last_idx = len(strides) - 1

        for i in range(last_idx):
            layers.append(
                block(self.inplanes, planes, strides[i], downsample, dilation))
            self.inplanes = planes * block.expansion
            downsample = None

        if non_local:
            layers.append(NLBlockND(in_channels=planes, mode='embedded',
                                    dimension=2, sub_sample=sub_sample, bn_layer=bn_layer))
            layers.append(
                block(self.inplanes, planes, strides[-1], dilation=dilation))
            layers.append(NLBlockND(in_channels=planes, mode='embedded',
                                    dimension=2, sub_sample=sub_sample, bn_layer=bn_layer))

        return nn.Sequential(*layers)


def resnet18(pretrained=True, extra_chan=0, non_local=False, sub_sample=False, bn_layer=False):
    if non_local:
        sub_sample = bn_layer = True
    model = ResNet(BasicBlock, [2, 2, 2, 2],
                   extra_chan, non_local, sub_sample, bn_layer)
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(
            model_urls['resnet18']), extra_chan)
    return model


def resnet34(pretrained=True, extra_chan=0, non_local=False, sub_sample=False, bn_layer=False):
    if non_local:
        sub_sample = bn_layer = True

    model = ResNet34(BasicBlock, [3, 4, 6, 3], non_local, sub_sample, bn_layer)

    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(
            model_urls['resnet34']), extra_chan)

    return model


def resnet50(pretrained=True, extra_chan=0, non_local=False, sub_sample=False, bn_layer=False):
    if non_local:
        sub_sample = bn_layer = True
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   extra_chan, non_local, sub_sample, bn_layer)
    if pretrained:
        load_weights_sequential(model, model_zoo.load_url(
            model_urls['resnet50']), extra_chan)
    return model
