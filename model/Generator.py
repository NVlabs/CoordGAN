# -------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2019 Kim Seonghyeon
# Copyright (c) 2021, Multimodal Lab @ Samsung AI Center Moscow
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

__all__ = ['CoordGAN',
           ]

import math
import random

import torch
from torch import nn

from .blocks import ConstantInput, StyledConv, ToRGB, PixelNorm, EqualLinear, LFF
import torch.nn.functional as F

def convert_to_coord_format(b, h, w, device='cpu', integer_values=False):
    if integer_values:
        x_channel = torch.arange(w, dtype=torch.float, device=device).view(1, 1, 1, -1).repeat(b, 1, w, 1)
        y_channel = torch.arange(h, dtype=torch.float, device=device).view(1, 1, -1, 1).repeat(b, 1, 1, h)
    else:
        x_channel = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, -1).repeat(b, 1, w, 1)
        y_channel = torch.linspace(-1, 1, h, device=device).view(1, 1, -1, 1).repeat(b, 1, 1, h)
    return torch.cat((x_channel, y_channel), dim=1)

class CoordGAN(nn.Module):
    def __init__(
        self,
        size,
        hidden_size, # fake input
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        activation=None,
        coord_outdim=2,
        **kwargs,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 512,
            128: 512,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4]+coord_outdim, style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channel = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            if i<=7:
                self.convs.append(
                    StyledConv(
                        in_channel+coord_outdim,
                        out_channel,
                        3,
                        style_dim,
                        #upsample=True,
                        blur_kernel=blur_kernel,
                    )
                )

                self.convs.append(
                    StyledConv(
                        out_channel+coord_outdim, out_channel, 3, style_dim, blur_kernel=blur_kernel
                    )
                )

                self.to_rgbs.append(ToRGB(out_channel+coord_outdim, style_dim, upsample=False))

            else:

                if i==8:
                    self.convs.append(
                        StyledConv(
                            in_channel+coord_outdim,
                            out_channel,
                            3,
                            style_dim,
                            upsample=True,
                            blur_kernel=blur_kernel,
                        )
                    )
                else:
                    self.convs.append(
                        StyledConv(
                            in_channel,
                            out_channel,
                            3,
                            style_dim,
                            upsample=True,
                            blur_kernel=blur_kernel,
                        )
                    )

                self.convs.append(
                    StyledConv(
                        out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                    )
                )

                self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

        self.lff = LFF(hidden_size)
        self.foldnet = CoordWarpNet(hidden_size+2, 2)

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.struc = nn.Sequential(*layers)


    def w_to_warp_coords(self, coords, struc):
        struc_lat = struc[0]

        coords = convert_to_coord_format(struc_lat.shape[0], 128, 128, struc_lat.device,
                                                integer_values=False)

        warp_coords = self.foldnet(coords, struc_lat)
#        warp_coords_lst = self.warp_coords(coords, struc_lat)
#        if len(warp_coords_lst)>1:
#            mid_warp_coords = warp_coords_lst[0]
#        warp_coords = warp_coords_lst[-1]
        return warp_coords

    def get_grid(self, F_size):
        b, c, h, w = F_size
        theta = torch.tensor([[1,0,0],[0,1,0]])
        theta = theta.unsqueeze(0).repeat(b,1,1)
        theta = theta.float()
        grid = torch.nn.functional.affine_grid(theta, F_size)
        return grid.cuda()

    def forward(
        self,
        style,
        struc,
        cur_size=128,
        alpha=1,
        mode=None,
        enc_lat_mode='w',
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):

        noise = [None] * self.num_layers

        struc = struc[0]
        style = style[0]
        if not input_is_latent:
            struc_w = self.struc(struc)
            style_w = self.style(style)
        else:
            struc_w = struc
            style_w = style

        if style_w.ndim < 3:
            latent = style_w.unsqueeze(1).repeat(1, self.n_latent, 1)
        else:
            latent = style_w

        coords = convert_to_coord_format(struc.shape[0], 128, 128, struc_w.device,
                                                integer_values=False)

        # warp coords
        warp_coords = self.foldnet(coords, struc_w)
        out = self.lff(warp_coords)
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        # grid = self.get_grid(out.shape)
        # warp_coords_sample = F.grid_sample(warp_coords, grid, align_corners=False, mode='nearest')
        out = torch.cat((out, warp_coords), dim=1)
        skip = self.to_rgb1(out, latent[:, 1])

        log_cur_size = int(math.log(cur_size, 2)) - 2
        skip_lst = []
        for i in range(0, log_cur_size*2, 2): 
            conv1 = self.convs[i]
            conv2 = self.convs[i+1]
            noise1 = noise[i+1]
            noise2 = noise[i+2]
            to_rgb = self.to_rgbs[i//2]

            out = conv1(out, latent[:, i+1], noise=noise1)
            if i<=8:
                out = torch.cat((out, warp_coords), dim=1)

            out = conv2(out, latent[:, i + 2], noise=noise2)
            if i<=8:
                out = torch.cat((out, warp_coords), dim=1)

            skip = to_rgb(out, latent[:, i + 3], skip)
            skip_lst.append(skip)

        image = skip
        if alpha<1:
            skip_image = skip_lst[-2]
            skip_image = F.interpolate(skip_image, scale_factor=2, mode='bilinear')
            out = (1 - alpha) * skip_image + alpha * image

        if mode=='vis':
            B, _, W, H = coords.shape
            warp_coords_vis = -torch.ones(B,3,W,H).to(coords.device)
            warp_coords_vis[:,:2,:,:] = warp_coords
            return image, warp_coords, warp_coords_vis
        elif mode=='mid_sup':
            return image, warp_coords, style_w, struc_w
        else:
            return image, None


class CoordWarpNet(torch.nn.Module):
    def __init__(self, in_ch=514, out_ch=2, lat_dim=512):
        super().__init__()

        self.conv1_1 = nn.Conv1d(in_ch, lat_dim, 1)
        self.conv1_2 = nn.Conv1d(lat_dim, lat_dim, 1)
        self.conv1_3 = nn.Conv1d(lat_dim, out_ch, 1)
        self.relu = nn.ReLU()

    def forward(self, coords, lat):
        B,C,H,W = coords.shape
        coords = coords.view(B,C,-1)
        lat = lat.unsqueeze(-1).repeat(1,1,H*W)

        x = torch.cat((coords, lat), dim=1)
        x = self.relu(self.conv1_1(x))  # x = batch,512,45^2
        x = self.relu(self.conv1_2(x))
        x = self.conv1_3(x)
        x = torch.tanh(x)

        out = x.view(B,2,H,W)

        return out
