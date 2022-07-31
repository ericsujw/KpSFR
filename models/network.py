"""
Modifed from STCN: https://github.com/hkchengrex/STCN
network.py - The core of the neural network
"""

import math
from matplotlib.pyplot import xscale

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import *


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


class Decoder(nn.Module):

    def __init__(self, model_archi):
        super().__init__()

        self.model_archi = model_archi

        self.n_classes = 1
        # self.n_classes = 92

        # part of encoder
        self.midconv_1 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.midconv_2 = nn.Sequential(
            nn.Upsample(size=(23, 40), mode='bilinear',
                        align_corners=True),

            nn.Conv2d(1024, 512, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.up_32_16 = self.up_conv(1024, 256, (45, 80))
        self.up_16_8 = self.up_conv(512, 128, (90, 160))
        self.up_8_4 = self.up_conv(256, 64, (180, 320))

        self.deconv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv_last = nn.Conv2d(64, 16, kernel_size=1)

        # Weights initialize
        self.midconv_1.apply(weights_init)
        self.midconv_2.apply(weights_init)
        self.up_32_16.apply(weights_init)
        self.up_16_8.apply(weights_init)
        self.up_8_4.apply(weights_init)
        self.deconv1.apply(weights_init)
        self.conv_last.apply(weights_init)

        # task-awareness of DoDNet, generate conv filters for classification layer
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.controller = nn.Conv2d(
            1024+91, 561, kernel_size=1, stride=1, padding=0)

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
            nn.ReLU(inplace=True)
        )

    def encoding_task(self, task_id):

        N = task_id.shape[0]
        task_encoding = torch.zeros(size=(N, 91), device=task_id.device)
        for i in range(N):
            if task_id[i] != -1:
                task_encoding[i, int(task_id[i]) - 1] = 1
        return task_encoding

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):

        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(
                    num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(
                    num_insts * 1, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 1)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):

        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def forward(self, f32, f16, f8, f4, cls):

        x = self.midconv_1(f32)

        task_encoding = self.encoding_task(cls)
        task_encoding.unsqueeze_(2).unsqueeze_(2)
        x_feat = self.GAP(x)
        x_cond = torch.cat([x_feat, task_encoding], dim=1)
        params = self.controller(x_cond)
        params.squeeze_(-1).squeeze_(-1)

        x = self.midconv_2(x)

        x = torch.cat([x, f32], dim=1)

        x = self.up_32_16(x)
        x = torch.cat([x, f16], dim=1)

        x = self.up_16_8(x)
        x = torch.cat([x, f8], dim=1)

        x = self.up_8_4(x)
        x = torch.cat([x, f4], dim=1)

        x = self.deconv1(x)
        x = self.conv_last(x)

        N, _, H, W = x.shape
        head_inputs = x.reshape(1, -1, H, W)

        weight_nums, bias_nums = [], []
        weight_nums.append(16*16)
        weight_nums.append(16*16)
        weight_nums.append(16*1)
        bias_nums.append(16)
        bias_nums.append(16)
        bias_nums.append(1)
        weights, biases = self.parse_dynamic_params(
            params, 16, weight_nums, bias_nums)

        logits = self.heads_forward(head_inputs, weights, biases, N)
        logits = logits.reshape(-1, 1, H, W)
        logits_origin = F.interpolate(
            logits, scale_factor=4, mode='bilinear', align_corners=False)

        return logits_origin


class KpSFR(nn.Module):

    def __init__(self, model_archi, num_objects, non_local):
        super(KpSFR, self).__init__()

        self.model_archi = model_archi

        self.key_encoder = KeyEncoder(
            num_objects=num_objects, non_local=non_local)

        # TODO: random pick 4
        self.value_encoder = ValueEncoder(
            num_objects=num_objects, non_local=non_local)

        # Projection from f32 feature space to key space
        self.key_proj = KeyProjection(512, keydim=64)

        # Compress f32 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.decoder = Decoder(model_archi=self.model_archi)

    def aggregate(self, prob):
        # values of prob is from (0, 1)
        assert prob.max() <= 1, print('out of value')
        assert prob.min() >= 0, print('out of value')
        new_prob = torch.cat([
            torch.prod(1 - prob, dim=1, keepdim=True),
            prob
        ], dim=1).clamp(1e-7, 1 - 1e-7)
        logits = torch.log((new_prob / (1 - new_prob)))
        return logits  # the inverse of sigmoid function

    def encode_key(self, frame, qcls=None):
        # input: b*t*c*h*w
        b, t = frame.shape[:2]

        f32, f16, f8, f4 = self.key_encoder(
            frame.flatten(start_dim=0, end_dim=1), qcls)
        assert torch.isnan(f32).sum() == 0, print('qf32: ', f32)
        assert torch.isnan(f16).sum() == 0, print('qf16: ', f16)
        assert torch.isnan(f8).sum() == 0, print('qf8: ', f8)
        assert torch.isnan(f4).sum() == 0, print('qf4: ', f4)

        # B*T*C*H*W
        f32 = f32.view(b, t, *f32.shape[-3:])
        f16 = f16.view(b, t, *f16.shape[-3:])
        f8 = f8.view(b, t, *f8.shape[-3:])
        f4 = f4.view(b, t, *f4.shape[-3:])

        return f32, f16, f8, f4

    def encode_value(self, frame, kf32, mask, other_masks=None, lookup=None, isFirst=False):
        # TODO: random pick 4
        # Extract memory key/value for a frame
        # input mask/other_masks: b*1*h*w, b*(objs-1)*h*w
        f32 = self.value_encoder(
            frame, kf32, mask, other_masks, lookup, isFirst)

        return f32.unsqueeze(2)  # B*256*T*H*W

    def segment(self, qf32, qf16, qf8, qf4, k, qcls, selector=None):
        # q - query, m - memory
        # qv32 is f32_thin above

        assert torch.isnan(qf32).sum() == 0, print('qf32: ', qf32)
        assert torch.isnan(qf16).sum() == 0, print('qf16: ', qf16)
        assert torch.isnan(qf8).sum() == 0, print('qf8: ', qf8)
        assert torch.isnan(qf4).sum() == 0, print('qf4: ', qf4)

        print('model archi: ', self.model_archi)
        x_origin = self.decoder(qf32, qf16, qf8, qf4, qcls[:, 0])

        for obj in range(1, k):
            x1_origin = self.decoder(qf32, qf16, qf8, qf4, qcls[:, obj])
            x_origin = torch.cat([x_origin, x1_origin], dim=1)

        assert torch.isnan(x_origin).sum() == 0, print('x_origin: ', x_origin)

        prob_origin = torch.sigmoid(x_origin)
        prob_origin = prob_origin * selector.unsqueeze(2).unsqueeze(2)
        logits_origin = self.aggregate(prob_origin)  # B*(obj+1)*720*1280
        prob_origin = F.softmax(logits_origin, dim=1)[:, 1:]  # B*obj*720*1280

        return x_origin, logits_origin, prob_origin

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError
