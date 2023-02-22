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


def get_cor_img(args, real, warp, temp=0.01, mode='aff', down_sample_ratio=1, num_cls=3):

    B,_,H,W = warp.shape
    pred_logits = F.interpolate(real, [H//down_sample_ratio,W//down_sample_ratio], mode='bilinear')
    if down_sample_ratio>1:
        warp = F.interpolate(warp, [H//down_sample_ratio,W//down_sample_ratio], mode='bilinear')

    assert warp.shape[0] % 2 == 0, "Minibatch size must be a multiple of 2"
    new_shape = [warp.shape[0] // 2, 2] + list(warp.shape[1:])
    warp = warp.view(*new_shape)

    new_shape = [pred_logits.shape[0] // 2, 2] + list(pred_logits.shape[1:])
    pred_logits = pred_logits.view(*new_shape) #(B//2, 2, C, H, W)
    B,N,C,H,W = pred_logits.shape
    pred_logits = pred_logits.permute(0,1,3,4,2).contiguous().view(B,N,-1,C)

    B,N,C,H,W = warp.shape
    warp = warp.permute(0,1,3,4,2).contiguous()\
                   .view(B,N,-1,C)
    diff = warp[:,0].unsqueeze(2) - warp[:,1].unsqueeze(1)
    diff = torch.norm(diff, dim=3)

    if mode=='aff':
        aff12 = F.softmax(- diff / temp, dim = 2)
        aff21 = F.softmax(- diff / temp, dim = 1).permute(0,2,1)
        #real12 = aff12@pred_logits[:,1].contiguous()
        #real21 = aff21@pred_logits[:,0].contiguous()
        real12 = torch.einsum('bik,bkj->bij', aff12, pred_logits[:,1].contiguous())
        real21 = torch.einsum('bik,bkj->bij', aff21, pred_logits[:,0].contiguous())

    real12 = real12.view(B,H,W,num_cls).permute(0,3,1,2).contiguous()
    real21 = real21.view(B,H,W,num_cls).permute(0,3,1,2).contiguous()

    _,C,H,W = real12.shape
    real12 = real12.unsqueeze(1)
    real21 = real21.unsqueeze(1)
    cor_img = torch.cat((real12, real21), dim=1)
    cor_img = cor_img.view(-1, C, H, W)

    return cor_img


