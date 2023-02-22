# ------------------------------------------------------------------------- 
# MIT License
#
# Copyright (c) 2019 Kim Seonghyeon
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Modified by Jiteng Mu
# ------------------------------------------------------------------------- 

__all__ = ['Discriminator']

import math

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import ConvLayer, ResBlock, EqualLinear


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], input_size=3, n_first_layers=0, **kwargs):
        super().__init__()

        self.input_size = input_size

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.from_rgb = nn.ModuleList(
            [
                ConvLayer(input_size, channels[128], 1),
                ConvLayer(input_size, channels[256], 1),
                ConvLayer(input_size, channels[512], 1),
            ])

        self.log_size = int(math.log(size, 2))

        in_channel = channels[size]

        convs = []
        for i in range(self.log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )


    def forward(self, input, cur_size=512, alpha=1):

        cur_log_size = int(math.log(cur_size, 2))
        skip_rgb_idx = int(math.log(cur_size//128, 2))
#        assert skip_rgb_idx>-1
        for i in range(self.log_size-cur_log_size, self.log_size-2):

            if i==self.log_size-cur_log_size:
                out = self.from_rgb[skip_rgb_idx](input)
#
#                if cur_size>128 and alpha<1:
#                    skip_rgb = F.avg_pool2d(input, 2)
#                    skip_rgb = self.from_rgb[skip_rgb_idx-1](skip_rgb)

            out = self.convs[i](out)

#            if i==self.log_size-cur_log_size and cur_size>128 and alpha<1:
#                out = (1 - alpha) * skip_rgb + alpha * out

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out
