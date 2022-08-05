'''
network for Nie et al. (A robust and efficient framework for sports-field registration)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import math

from models.non_local import NLBlockND


def weights_init(m):
    # Initialize filters with Gaussian random weights
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


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_planes * self.expansion),
            )

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


class EncDec(nn.Module):

    def up_conv(self, in_channels, out_channels, size):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels //
                      2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),

            nn.Upsample(size=size, mode='bilinear', align_corners=True),

            nn.Conv2d(in_channels // 2, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def __init__(self, layers, n_classes, non_local, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError(
                'Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        if non_local:
            sub_sample = bn_layer = True

        super(EncDec, self).__init__()
        pretrained_model = torchvision.models.__dict__[
            'resnet{}'.format(layers)](pretrained=pretrained)

        self.in_planes = 256
        self.conv1 = pretrained_model._modules['conv1']
        self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']

        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        # self.layer4 = pretrained_model._modules['layer4']

        # TODO add a few dilated convolution and non-local block layers
        self.layer4 = self._make_layer(
            BasicBlock, 512, 2, stride=2, dilation=2, non_local=non_local, sub_sample=sub_sample, bn_layer=bn_layer)
        self.layer4[0].apply(weights_init)
        self.layer4[2].apply(weights_init)
        self.layer4[1].apply(weights_init2)
        self.layer4[3].apply(weights_init2)

        # Clear memory
        del pretrained_model

        self.midconv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.Upsample(size=(23, 40), mode='bilinear', align_corners=True),

            nn.Conv2d(1024, 512, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.up3 = self.up_conv(1024, 256, (45, 80))
        self.up2 = self.up_conv(512, 128, (90, 160))
        self.up1 = self.up_conv(256, 64, (180, 320))

        self.conv_last = nn.Conv2d(128, n_classes, kernel_size=1)

        # Weights initialize
        self.midconv.apply(weights_init)
        self.up3.apply(weights_init)
        self.up2.apply(weights_init)
        self.up1.apply(weights_init)
        self.conv_last.apply(weights_init)

    def _make_layer(self, block, out_planes, num_blocks, stride, dilation=1, non_local=False, sub_sample=False, bn_layer=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        last_idx = len(strides)

        if non_local:
            last_idx = len(strides) - 1

        for i in range(last_idx):
            layers.append(
                block(self.in_planes, out_planes, strides[i], dilation))
            self.in_planes = out_planes * block.expansion

        if non_local:
            layers.append(NLBlockND(in_channels=out_planes, mode='embedded',
                                    dimension=2, sub_sample=sub_sample, bn_layer=bn_layer))
            layers.append(
                block(self.in_planes, out_planes, strides[-1], dilation))
            layers.append(NLBlockND(in_channels=out_planes, mode='embedded',
                                    dimension=2, sub_sample=sub_sample, bn_layer=bn_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # === resnet ===
        [bs, c, h, w] = x.size()

        x_original = self.conv1(x)
        x_original = self.bn1(x_original)
        x_original = self.relu(x_original)

        conv0 = self.maxpool(x_original)
        conv1 = self.layer1(conv0)  # size=(N, 64, x.H/4, x.W/4)
        conv2 = self.layer2(conv1)  # size=(N, 128, x.H/8, x.W/8)
        conv3 = self.layer3(conv2)  # size=(N, 256, x.H/16, x.W/16)
        conv4 = self.layer4(conv3)  # size=(N, 512, x.H/32, x.W/32)

        # === decoder ===
        x = self.midconv(conv4)  # 512
        x = torch.cat([x, conv4], dim=1)
        x = self.up3(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.up2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.up1(x)
        x = torch.cat([x, conv1], dim=1)

        out = self.conv_last(x)

        return out
