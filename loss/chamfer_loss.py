# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# To view a copy of this license, visit
# https://github.com/NVlabs/CoordGAN/blob/main/LICENSE
#
# Written by Jiteng Mu
# -------------------------------------------------------------------------

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.loss import chamfer_distance

def warp_chamfer_loss(warp_coords, std_coords):
    B,C,H,W = warp_coords.shape
    if std_coords.shape[-1]!=W:
        std_coords = F.interpolate(std_coords, [H,W])
    std_coords_flat = std_coords.permute(0,2,3,1).view(B,H*W,2)
    warp_coords_flat = warp_coords.permute(0,2,3,1).view(B,H*W,2)
    loss, _ = chamfer_distance(warp_coords_flat, std_coords_flat)
    return loss


