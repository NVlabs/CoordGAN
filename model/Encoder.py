# -------------------------------------------------------------------------
# Creative Commons — Attribution-NonCommercial-ShareAlike 4.0 International — CC BY-NC-SA 4.0
# See CC-BY-NC-SA-4.0.md for a human-readable summary of (and not a substitute for) the license.
#
# Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way  that suggests the licensor endorses you or your use.
#
# NonCommercial — You may not use the material for commercial purposes.
#
# ShareAlike — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.
#
# No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.
#
# Modified by Jiteng Mu
# -------------------------------------------------------------------------


import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from .blocks import ConvLayer_swap, ToRGB, EqualLinear, Blur, Upsample, make_kernel
import utils


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], reflection_pad=False, pad=None, downsample=True):
        super().__init__()

        self.conv1 = ConvLayer_swap(in_channel, in_channel, 3, reflection_pad=reflection_pad, pad=pad)
        self.conv2 = ConvLayer_swap(in_channel, out_channel, 3, downsample=downsample, blur_kernel=blur_kernel, reflection_pad=reflection_pad, pad=pad)

        self.skip = ConvLayer_swap(
            in_channel, out_channel, 1, downsample=downsample, blur_kernel=blur_kernel, activate=False, bias=False
        )

    def forward(self, input):
        #print("before first resnet layeer, ", input.shape)
        out = self.conv1(input)
        #print("after first resnet layer, ", out.shape)
        out = self.conv2(out)
        #print("after second resnet layer, ", out.shape)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class ToSpatialCode(torch.nn.Module):
    def __init__(self, inch, outch, scale):
        super().__init__()
        hiddench = inch // 2
        self.conv1 = ConvLayer_swap(inch, hiddench, 1, activate=True, bias=True)
        self.conv2 = ConvLayer_swap(hiddench, outch, 1, activate=False, bias=True)
        self.scale = scale
        self.upsample = Upsample([1, 3, 3, 1], 2)
        self.blur = Blur([1, 3, 3, 1], pad=(2, 1))
        self.register_buffer('kernel', make_kernel([1, 3, 3, 1]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        for i in range(int(np.log2(self.scale))):
            x = self.upsample(x)
        return x


class Encoder(torch.nn.Module):
#    @staticmethod
#    def modify_commandline_options(parser, is_train):
#        parser.add_argument("--netE_scale_capacity", default=1.0, type=float)
#        parser.add_argument("--netE_num_downsampling_sp", default=4, type=int)
#        parser.add_argument("--netE_num_downsampling_gl", default=2, type=int)
#        parser.add_argument("--netE_nc_steepness", default=2.0, type=float)
#        return parser

    def __init__(self, opt):
        super().__init__()
        self.netE_scale_capacity = 1
        opt.netE_num_downsampling_sp = 4
        opt.netE_num_downsampling_gl = 2
        self.netE_nc_steepness = 2
        opt.global_code_ch = 512
        opt.spatial_code_ch = 64
        opt.use_antialias = True
        self.sp_fc = EqualLinear(8*8*64, 512)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # If antialiasing is used, create a very lightweight Gaussian kernel.
        blur_kernel = [1, 2, 1] if opt.use_antialias else [1]

        self.add_module("FromRGB", ConvLayer_swap(3, self.nc(0), 1))

        self.DownToSpatialCode = nn.Sequential()
        for i in range(opt.netE_num_downsampling_sp):
            self.DownToSpatialCode.add_module(
                "ResBlockDownBy%d" % (2 ** i),
                ResBlock(self.nc(i), self.nc(i + 1), blur_kernel,
                         reflection_pad=True)
            )

        # Spatial Code refers to the Structure Code, and
        # Global Code refers to the Texture Code of the paper.
        nchannels = self.nc(opt.netE_num_downsampling_sp)
        self.add_module(
            "ToSpatialCode",
            nn.Sequential(
                ConvLayer_swap(nchannels, nchannels, 1, activate=True, bias=True),
                ConvLayer_swap(nchannels, opt.spatial_code_ch, kernel_size=1,
                          activate=False, bias=True, pad=0)
            )
        )

        self.DownToGlobalCode = nn.Sequential()
        for i in range(opt.netE_num_downsampling_gl):
            idx_from_beginning = opt.netE_num_downsampling_sp + i
            self.DownToGlobalCode.add_module(
                "ConvLayerDownBy%d" % (2 ** idx_from_beginning),
                ConvLayer_swap(self.nc(idx_from_beginning),
                          self.nc(idx_from_beginning + 1), kernel_size=3,
                          blur_kernel=[1], downsample=True, pad=1)
            )

        nchannels = self.nc(opt.netE_num_downsampling_sp +
                            opt.netE_num_downsampling_gl)
#        self.add_module(
#            "ToGlobalCode",
#            nn.Sequential(
#                EqualLinear(nchannels, opt.global_code_ch)
#            )
#        )

        self.ToGlobalCode = nn.Sequential()
        for i in range(12):
            self.ToGlobalCode.add_module(
                "%d" % (i),
                    EqualLinear(nchannels, opt.global_code_ch)
            )


    def nc(self, idx):
        nc = self.netE_nc_steepness ** (5 + idx)
        nc = nc * self.netE_scale_capacity
        # nc = min(self.opt.global_code_ch, int(round(nc)))
        return round(nc)

    def kl_divergence(self, mu, logvar):
        #kl_q_z_p_z = 0.5 * torch.sum(-2 * torch.log(sigma) - 1 + sigma**2 + mu**2)
        kl_q_z_p_z = 0.5 * torch.sum(-logvar - 1 + torch.exp(logvar) + mu**2)
        return kl_q_z_p_z

    def rsample(self, mu, logvar):
        std_norm = torch.randn_like(mu).to(mu.device)
        std = torch.exp(0.5 * logvar)
        samples = std * std_norm + mu
        return samples

    def forward(self, x, extract_features=False):
        x = self.FromRGB(x)
        midpoint = self.DownToSpatialCode(x)
        sp = self.ToSpatialCode(midpoint)
        b, _, _, _ = sp.shape
        sp = sp.view(b,-1)
        sp = self.sp_fc(sp)
        # sp = utils.ch_norm(sp)

        if extract_features:
            padded_midpoint = F.pad(midpoint, (1, 0, 1, 0), mode='reflect')
            feature = self.DownToGlobalCode[0](padded_midpoint)
            assert feature.size(2) == sp.size(2) // 2 and \
                feature.size(3) == sp.size(3) // 2
            feature = F.interpolate(
                feature, size=(7, 7), mode='bilinear', align_corners=False)

        x = self.DownToGlobalCode(midpoint)
        # x = F.instance_norm(x)
        #x = x.mean(dim=(2, 3))
        x = self.avgpool(x).squeeze(-1).squeeze(-1)
        all_gl = []
        for i in range(12):
            all_gl.append(self.ToGlobalCode[i](x).unsqueeze(1))
        gl = torch.cat(all_gl, dim=1)
        # gl = utils.ch_norm(gl)
        #sp = util.normalize(sp)
        #gl = util.normalize(gl)

        if extract_features:
            return sp, gl, feature
        else:
            return sp, gl, _
