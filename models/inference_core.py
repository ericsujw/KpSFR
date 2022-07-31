"""
Modifed from STCN: https://github.com/hkchengrex/STCN
"""

import torch
import torch.nn.functional as F

from models.eval_network import EvalKpSFR

import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import utils


class InferenceCore:

    def __init__(self, prop_net: EvalKpSFR, images, device, num_objects, lookup=None):

        self.prop_net = prop_net

        # True dimensions
        t = images.shape[1]
        h, w = images.shape[-2:]

        nh = images.shape[-2] // 4
        nw = images.shape[-1] // 4

        self.images = images
        self.device = device

        self.k = num_objects
        self.lookup = lookup

        # Background included, not always consistent (i.e. sum up to 1)
        self.prob = torch.zeros(
            (self.k + 1, t, 1, nh, nw), dtype=torch.float32, device=self.device)
        self.prob[0] = 1e-7

        self.t, self.h, self.w = t, h, w
        self.nh, self.nw = nh, nw

    def aggregate(self, prob, keep_bg=False):
        # values of prob is from (0, 1)
        assert prob.max() <= 1, print('out of value')
        assert prob.min() >= 0, print('out of value')
        k = prob.shape
        new_prob = torch.cat([
            torch.prod(1 - prob, dim=0, keepdim=True),
            prob
        ], dim=0).clamp(1e-7, 1 - 1e-7)
        logits = torch.log((new_prob / (1 - new_prob)))

        if keep_bg:
            return logits, F.softmax(logits, dim=0)
        else:
            return logits, F.softmax(logits, dim=0)[1:]

    def encode_key(self, idx):

        result = self.prop_net.encode_key(self.images[:, idx])
        return result

    def do_pass(self, idx, end_idx, selector):

        print(f'Current frame is {idx}')

        closest_ti = end_idx

        # Note that we never reach closest_ti, just the frame before it
        print('Start frame: ', idx)
        this_range = range(idx, closest_ti)
        end = closest_ti - 1

        for ti in this_range:
            print(f'Current frame is {ti}')
            qf32, qf16, qf8, qf4 = self.encode_key(ti)
            out_mask_origin = self.prop_net.segment_with_query(
                self.k, qf32, qf16, qf8, qf4, self.lookup[ti])

            _, out_prob_origin = self.aggregate(
                out_mask_origin, keep_bg=True)
            out_prob = F.interpolate(
                out_prob_origin, (self.nh, self.nw), mode='bilinear', align_corners=False)
            self.prob[:, ti] = out_prob

        return closest_ti

    def interact(self, frame_idx, end_idx, selector=None):

        # Propagate
        self.do_pass(frame_idx, end_idx, selector)
