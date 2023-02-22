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

"""https://github.com/taesungp/swapping-autoencoder-pytorch/blob/76879fdaa4a4e8e49b3881e6e4691dbcae438992/models/networks/patch_discriminator.py"""

from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#import util
#from models.networks import BaseNetwork
from .blocks import ConvLayer_swap, ResBlock_swap, EqualLinear


class PatchDiscriminator(nn.Module):

    def __init__(self, args):
        super().__init__()
        channel_multiplier = args.patch_scale_capacity # ?
        netPatchD_max_nc = args.patch_max_nc  # ?
        size = args.patch_size # ?
        channels = {
            4: min(netPatchD_max_nc, int(256 * channel_multiplier)),
            8: min(netPatchD_max_nc, int(128 * channel_multiplier)),
            16: min(netPatchD_max_nc, int(64 * channel_multiplier)),
            32: int(32 * channel_multiplier),
            64: int(16 * channel_multiplier),
            128: int(8 * channel_multiplier),
            256: int(4 * channel_multiplier),
        }

        log_size = int(math.ceil(math.log(size, 2)))

        in_channel = channels[2 ** log_size]

        blur_kernel = [1, 3, 3, 1]

        convs = [('0', ConvLayer_swap(3, in_channel, 3))]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            layer_name = str(7 - i) if i <= 6 else "%dx%d" % (2 ** i, 2 ** i)
            convs.append((layer_name, ResBlock_swap(in_channel, out_channel, blur_kernel)))

            in_channel = out_channel

        convs.append(('5', ConvLayer_swap(netPatchD_max_nc, netPatchD_max_nc, 3, pad=0)))

        self.convs = nn.Sequential(OrderedDict(convs))

        out_dim = 1

        pairlinear1 = EqualLinear(channels[4] * 2 * 2 * 2, 1024, activation='fused_lrelu')
        pairlinear2 = EqualLinear(1024, out_dim)
        self.pairlinear = nn.Sequential(pairlinear1, pairlinear2)

    def extract_features(self, patches, aggregate=False):
        if patches.ndim == 5:
            B, T, C, H, W = patches.size()
            flattened_patches = patches.flatten(0, 1)
        else:
            B, C, H, W = patches.size()
            T = patches.size(1)
            flattened_patches = patches
        features = self.convs(flattened_patches)
        features = features.view(B, T, features.size(1), features.size(2), features.size(3))
        if aggregate:
            features = features.mean(1, keepdim=True).expand(-1, T, -1, -1, -1)
        return features.flatten(0, 1)

    def extract_layerwise_features(self, image):
        feats = [image]
        for m in self.convs:
            feats.append(m(feats[-1]))

        return feats

    def discriminate_features(self, feature1, feature2):
        feature1 = feature1.flatten(1)
        feature2 = feature2.flatten(1)
        out = self.pairlinear(torch.cat([feature1, feature2], dim=1))
        return out

