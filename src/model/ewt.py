from model import common
# import common
import torch
import torch.nn as nn
import torch.nn.functional as F
# import scipy.io as sio
from model.MFAM import MFAM
from MFAM import MFAM
import os

def make_model(args, parent=False):
    return EWT(args)

class EWT(nn.Module):
    # def __init__(self, args, conv=common.default_conv):
    def __init__(self, conv=common.default_conv):
        super(EWT, self).__init__()
        self.scale_idx = 0

        self.DWT = common.DWT()
        self.IWT = common.IWT()
        self.trans = MFAM(upscale=1, img_size=(32, 32), in_chans=12,
                      window_size=8, img_range=1., depths=[6, 6, 6, 6],
                      embed_dim=180, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='')


    def _padding(self, x, scale):
        delta_H = 0
        delta_W = 0
        if x.shape[2] % scale != 0:
            delta_H = scale - x.shape[2] % scale
            x = F.pad(x, (0, 0, 0, delta_H), 'reflect')
        if x.shape[3] % scale != 0:
            delta_W = scale - x.shape[3] % scale
            x = F.pad(x, (0, delta_W, 0, 0), 'reflect')
        return x, delta_H, delta_W

    def _padding_2(self, x):
        _, _, H, W = x.shape
        delta = abs(H-W)
        if H < W:
            x = F.pad(x, (0, 0, 0, delta), 'reflect')
        elif H > W:
            x = F.pad(x, (0, delta, 0, 0), 'reflect')
        return x


    def forward(self, x):
        _, _, H, W = x.shape
        # x = self._padding_2(x)
        x, delta_H, delta_W = self._padding(x, 8)

        x = self.DWT(x)
        x = self.trans(x)
        x = self.IWT(x)

        x = x[:, :, :H, :W]

        return x