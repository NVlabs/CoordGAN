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

import os
import utils


def visualization(args, sample_z, g_ema, converted_full, cur_size, alpha, device, real_img=None, enc_input_is_latent=False):

    cur_z_style = torch.randn(args.n_sample, args.latent, device=device)
    cur_z_struc = torch.randn(args.n_sample, args.latent, device=device)
    if real_img is not None:
        cur_z_style, cur_z_struc = sample_z

    real_cur, warp_cur, vis_cur = g_ema([cur_z_style], [cur_z_struc], cur_size, alpha, mode='vis', input_is_latent=enc_input_is_latent)

    if real_cur.shape[-1]!=vis_cur.shape[-1]:
        vis_cur = F.interpolate(vis_cur, [real_cur.shape[-2], real_cur.shape[-1]])

    sample_cur = real_cur.unsqueeze(1)
    vis_cur = vis_cur.unsqueeze(1)
    sample_cur = torch.cat((vis_cur, sample_cur), dim=1)
    if real_img is not None:
        sample_cur = torch.cat((real_img.unsqueeze(1), sample_cur), dim=1)

    swap_struc=True
    if swap_struc==True:
        swap_style_cur = utils.swap([cur_z_style])
        swap_sample_cur, swap_warp_cur, swap_vis_cur = g_ema([swap_style_cur], [cur_z_struc], cur_size, alpha, mode='vis', input_is_latent=enc_input_is_latent)

        swap_vis_cur = F.interpolate(swap_vis_cur, [real_cur.shape[-2], real_cur.shape[-1]])
        swap_sample_cur = F.interpolate(swap_sample_cur, [real_cur.shape[-2], real_cur.shape[-1]])

        sample_cur = torch.cat((sample_cur, swap_vis_cur.unsqueeze(1)), dim=1)
        sample_cur = torch.cat((sample_cur, swap_sample_cur.unsqueeze(1)), dim=1)


    img_transfer = True
    if img_transfer==True:

        real1, real2, real12, real21 = get_cor_real(args, real_cur, warp_cur, temp=args.rgb_temp, mode=args.rgb_mode, down_sample_ratio=args.rgb_down_sample_ratio)

        _,C,H,W = real_cur.shape
        if real12.shape[-1]!=real_cur.shape[-1]:
            real1 = F.interpolate(real1, [H, W], mode='bilinear')
            real2 = F.interpolate(real2, [H, W], mode='bilinear')
            real12 = F.interpolate(real12, [H, W], mode='bilinear')
            real21 = F.interpolate(real21, [H, W], mode='bilinear')

        real_12 = torch.cat((real1.unsqueeze(1), real21.unsqueeze(1)), dim=1 ).view(-1,C,H,W)
        real_21 = torch.cat((real12.unsqueeze(1), real2.unsqueeze(1)), dim=1 ).view(-1,C,H,W)
        sample_cur = torch.cat((sample_cur, real_12.unsqueeze(1)), dim=1)
        sample_cur = torch.cat((sample_cur, real_21.unsqueeze(1)), dim=1)

    sample_cur = sample_cur.view(-1, 3, cur_size, cur_size)

    return sample_cur


def get_cor_real(args, real, warp, temp=0.01, mode='aff', down_sample_ratio=4, seg_model=None, num_cls=3):

    if seg_model is not None:
        if args.dataset=='ffhq':
            real_ = F.interpolate(real.clone(), [512, 512])
            seg_logits = seg_model(real_)["out"]
            seg_logits = F.interpolate(seg_logits, [128, 128])
            seg = torch.argmax(seg_logits, dim=1)
            seg[seg==17] = 0
        else:
            seg_logits = seg_model(real)["out"]
            seg = torch.argmax(seg_logits, dim=1)

        seg = seg.detach()
        seg[seg>0] = 1
        real = torch.mul(real, seg.unsqueeze(1))

    if down_sample_ratio>1:
        B,_,H,W = warp.shape
        pred_logits = F.interpolate(real, [H//down_sample_ratio,W//down_sample_ratio], mode='nearest')
        pred = F.interpolate(real, [H//down_sample_ratio,W//down_sample_ratio], mode='nearest')
        warp = F.interpolate(warp, [H//down_sample_ratio,W//down_sample_ratio], mode='nearest')
    else:
        pred_logits = real.clone()
        pred = real.clone()

    _,_,H,W = pred.shape
    if W != warp.shape[-1]:
        ratio = W//warp.shape[-1]
        pred_logits = F.interpolate(pred_logits, [H//ratio,W//ratio], mode='bilinear')
        pred = F.interpolate(pred, [H//ratio,W//ratio], mode='bilinear')

    assert warp.shape[0] % 2 == 0, "Minibatch size must be a multiple of 2"
    new_shape = [warp.shape[0] // 2, 2] + list(warp.shape[1:])
    warp = warp.view(*new_shape)

    new_shape = [pred.shape[0] // 2, 2] + list(pred.shape[1:])
    pred = pred.view(*new_shape)

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
        real12 = torch.einsum('bik,bkj->bij', aff12, pred_logits[:,1].contiguous())
        real21 = torch.einsum('bik,bkj->bij', aff21, pred_logits[:,0].contiguous())
    if mode=='gumbel':
        aff12 = F.gumbel_softmax(-diff, tau=temp, dim=2, hard=True)
        aff21 = F.gumbel_softmax(-diff.permute(0,2,1), tau=temp, dim=2, hard=True)
        real12 = torch.einsum('bik,bkj->bij', aff12, pred_logits[:,1].contiguous())
        real21 = torch.einsum('bik,bkj->bij', aff21, pred_logits[:,0].contiguous())

    real12 = real12.view(B,H,W,num_cls).permute(0,3,1,2).contiguous()
    real21 = real21.view(B,H,W,num_cls).permute(0,3,1,2).contiguous()

    pred1 = pred[:,0] 
    pred2 = pred[:,1] 

    return pred1, pred2, real12, real21

