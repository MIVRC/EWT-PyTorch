from model import common
# import common
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.SwinIR_simple import SwinIR
# from SwinIR_simple import SwinIR


def make_model(args, parent=False):
    return MWT(args)


def default_conv(in_channels=None, out_channels=None, kernel_size=None, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class MWT(nn.Module):
    def __init__(self, conv=default_conv):
        super(MWT, self).__init__()
        # print("MWT")
        kernel_size = 3
        self.scale_idx = 0

        act = nn.ReLU(True)

        self.DWT = common.DWT()
        self.IWT = common.IWT()

        d_l1 = [common.BBlock(conv, 3 * 4, 64, kernel_size, act=act, bn=False)]
        d_l1.append(common.BBlock(conv, 64, 64, kernel_size, act=act, bn=False))

        d_l2 = []
        d_l2.append(common.BBlock(conv, 64 * 4, 128, kernel_size, act=act, bn=False))
        d_l2.append(common.BBlock(conv, 128, 128, kernel_size, act=act, bn=False))

        d_l3 = []
        d_l3.append(common.BBlock(conv, 128 * 4, 128, kernel_size, act=act, bn=False))
        d_l3.append(common.BBlock(conv, 128, 128, kernel_size, act=act, bn=False))
        # d_l3.append(common.BBlock(conv, 128, 64, kernel_size, act=act, bn=False))

        i_l3 = [common.BBlock(conv, 128, 128 * 4, kernel_size, act=act, bn=False)]

        i_l2 = [common.BBlock(conv, 128, 128, kernel_size, act=act, bn=False)]
        # i_l2.append(common.BBlock(conv, 128, 128, kernel_size, act=act, bn=False))
        i_l2.append(common.BBlock(conv, 128, 128 * 2, kernel_size, act=act, bn=False))

        i_l1 = [common.BBlock(conv, 64, 64, kernel_size, act=act, bn=False)]
        # i_l1.append(common.BBlock(conv, 64, 64, kernel_size, act=act, bn=False))
        i_l1.append(common.BBlock(conv, 64, 12, kernel_size, act=act, bn=False))

        self.d_l1 = nn.Sequential(*d_l1)
        self.d_l2 = nn.Sequential(*d_l2)
        self.trans_2 = SwinIR(in_chans=128, embed_dim=192, depths=[2, 4, 8, 8], num_heads=[2, 2, 4, 4])
        # self.trans_2 = nn.Sequential(*d_l3)
        self.d_l3 = nn.Sequential(*d_l3)
        self.trans_3 = SwinIR(in_chans=128, embed_dim=192, depths=[2, 2, 6, 4], num_heads=[3, 3, 6, 3])
        self.i_l3 = nn.Sequential(*i_l3)
        self.i_l2 = nn.Sequential(*i_l2)
        self.i_l1 = nn.Sequential(*i_l1)

        # self.conv1 = nn.Conv2d(48, 64, 3, 1, 1)
        # self.conv2 = nn.Conv2d(192, 128, 3, 1, 1)

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

    def forward(self, x):
        _, _, H, W = x.shape
        #
        x, delta_H, delta_W = self._padding(x, 8)
        # shortcut = x
        x = self.DWT(x)
        x = self.d_l1(x)    # 64
        x_s1 = x

        x = self.DWT(x)
        x = self.d_l2(x) # 128
        x_s2 = x
        # x_ = self.trans_2(x)

        x = self.DWT(x)
        x = self.d_l3(x)    #
        # x = self.trans_3(x)

        x = self.i_l3(x)
        x = self.IWT(x)

        x = self.i_l2(x + x_s2)
        x = self.IWT(x)

        x = self.i_l1(x + x_s1)
        x = self.IWT(x)

        x = x[:, :, :H, :W]
        return x


