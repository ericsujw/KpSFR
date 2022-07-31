"""
Modifed from STCN: https://github.com/hkchengrex/STCN
eval_network.py - Evaluation version of the network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import *
from models.network import Decoder

import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import utils


class EvalKpSFR(nn.Module):

    def __init__(self, model_archi, num_objects, non_local):
        super().__init__()

        self.k = num_objects
        self.model_archi = model_archi

        self.key_encoder = KeyEncoder(
            num_objects=0, non_local=non_local)

        # TODO: random pick 4
        self.value_encoder = ValueEncoder(
            num_objects=num_objects, non_local=non_local)

        # Projection from f32 feature space to key space
        self.key_proj = KeyProjection(512, keydim=64)

        # Compress f32 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.decoder = Decoder(model_archi=self.model_archi)

    def encode_key(self, frame):
        # frame: b*c*h*w

        f32, f16, f8, f4 = self.key_encoder(frame, None)

        return f32, f16, f8, f4

    def segment_with_query(self, k, qf32, qf16, qf8, qf4, lookup):

        print('model archi: ', self.model_archi)
        qf32 = qf32.expand(k, -1, -1, -1)
        qf16 = qf16.expand(k, -1, -1, -1)
        qf8 = qf8.expand(k, -1, -1, -1)
        qf4 = qf4.expand(k, -1, -1, -1)

        x_origin = self.decoder(
            qf32[0:1], qf16[0:1], qf8[0:1], qf4[0:1], lookup[0:1])
        for idx in range(1, k):
            x1_origin = self.decoder(
                qf32[idx:idx+1], qf16[idx:idx+1], qf8[idx:idx+1], qf4[idx:idx+1], lookup[idx:idx+1])
            x_origin = torch.cat([x_origin, x1_origin], dim=0)

        return torch.sigmoid(x_origin)
